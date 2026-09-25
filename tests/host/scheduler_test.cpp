// M6 Stage 2b: the scheduler policy gate. A recording FakeEngine stands in
// for GlmDiagnosticModel + the pick, so every policy decision is pinned
// WITHOUT a GPU, a model cache, or the bus — this gate always runs.
//
// What is pinned here (the op stream is the contract):
//   * strict alternation — at most one admission, exactly one decode step
//     per tick, admission first;
//   * FCFS admission without head-of-line blocking, budget-deferred
//     until a peer retires, and the loud admission deadlock;
//   * round-robin decode by arrival order, slot reuse (lowest free);
//   * EOS on the prefill pick and mid-run; the steps cap;
//   * scripted cancellation, and THE ISOLATION PROPERTY: a request's
//     generated ids are identical whether it runs alone or interleaved
//     with (and cancelled around) other requests;
//   * determinism — the same scenario produces the identical op stream.
//
// The fake also enforces the engine interface: scripts are consumed in order,
// each slot owns its pending transcript, and close() must find a live
// reservation to release.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

#include "common/log.hpp"

#include "common/test.hpp"
#include "sched/scheduler.hpp"
#include "serve/prefill_policy.hpp"

namespace {

using dgpp::sched::Scheduler;
using dgpp::sched::SchedulerEngine;
using dgpp::sched::SchedulerRequest;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// The recording engine: scripted token episodes per slot (episodes queue,
// so a retired slot's next admission consumes the next episode), mirrored
// block reservations, and the op stream the tests pin.
class FakeEngine : public SchedulerEngine {
 public:
  FakeEngine(int slots, int64_t total_blocks, int64_t block_tokens,
             int batch_capacity = 1, bool can_sample = false)
      : slots_(slots),
        total_blocks_(total_blocks),
        block_tokens_(block_tokens),
        batch_capacity_(batch_capacity),
        can_sample_(can_sample) {}

  // The sampling interface (M6 6b): a sampling-capable fake records the arming
  // in the op stream ("A:slot:seed"), so its position relative to the
  // prefill is pinned; a greedy fake inherits the base refusal.
  bool supports_sampling() const override { return can_sample_; }
  void configure_sampling(int req, const dgpp::sample::Params& p,
                          uint64_t seed) override {
    if (!can_sample_) {
      SchedulerEngine::configure_sampling(req, p, seed);
      return;
    }
    if (p.temperature > 0.0f)
      ops_.push_back("A:" + std::to_string(req) + ":" + std::to_string(seed));
  }
  // Logprobs: a sampling-capable fake reports one entry per token it
  // returns (logprob = -token, one alternative), only for armed slots.
  bool supports_logprobs() const override { return can_sample_; }
  void configure_logprobs(int req, int logprobs) override {
    if (!can_sample_) {
      SchedulerEngine::configure_logprobs(req, logprobs);
      return;
    }
    report_[req] = logprobs;
    if (logprobs >= 0) ops_.push_back("L:" + std::to_string(req));
  }
  std::vector<dgpp::sample::Result> take_logprobs(int req) override {
    std::vector<dgpp::sample::Result> out;
    out.swap(pending_lps_[req]);
    return out;
  }
  // Constrained decoding (M6 6g): a sampling-capable fake can mask and
  // records the grammar arming ("G:slot:mode:tools") right after the
  // sampling arming; a greedy fake inherits the base refusal.
  bool supports_constraints() const override { return can_sample_; }
  void configure_constraint(int req,
                            const dgpp::text::GrammarSpec& g) override {
    if (!can_sample_) {
      SchedulerEngine::configure_constraint(req, g);
      return;
    }
    if (g.active())
      ops_.push_back("G:" + std::to_string(req) + ":" +
                     std::to_string(static_cast<int>(g.mode)) + ":" +
                     std::to_string(g.tools.size()));
  }
  // The logit bias: a sampling-capable fake can bias and
  // records the arming ("B:slot:entries") after the grammar's.
  bool supports_logit_bias() const override { return can_sample_; }
  void configure_logit_bias(int req,
                            const std::vector<dgpp::sched::LogitBias>& b) override {
    if (!can_sample_) {
      SchedulerEngine::configure_logit_bias(req, b);
      return;
    }
    if (!b.empty())
      ops_.push_back("B:" + std::to_string(req) + ":" + std::to_string(b.size()));
  }

  // Arms `slot`'s NEXT scalar episode: prefill returns tokens[0], then each
  // step returns one following token.
  void arm(int slot, std::vector<int32_t> tokens, int max_steps) {
    require(!tokens.empty(), "fake: cannot arm an empty token script");
    Episode e;
    e.first = tokens.front();
    e.max_steps = max_steps;
    for (size_t i = 1; i < tokens.size(); ++i)
      e.steps.push_back({tokens[i]});
    episodes_[slot].push_back(std::move(e));
  }

  // Arms the new speculative interface directly: prefill returns `first`, and
  // each subsequent engine step returns one scripted batch.
  void arm_batches(int slot, int32_t first,
                   std::vector<std::vector<int32_t>> steps, int max_steps) {
    episodes_[slot].push_back(Episode{first, std::move(steps), max_steps});
  }

  const std::vector<std::string>& ops() const { return ops_; }
  const std::vector<std::vector<int>>& batch_calls() const {
    return batch_calls_;
  }
  std::string op_stream() const {
    std::string s;
    for (const std::string& op : ops_) {
      if (!s.empty()) s += " ";
      s += op;
    }
    return s;
  }

  // The prefix cache interface (M7): an arena of `slots` snapshot slots with
  // pool alignment `align`. The fake records every op — "X:slot:pos@arena"
  // an attach, "N:slot:pos@arena" a snapshot taken at a prefill cut,
  // "RS:slot:pos@arena" a rolling snapshot, "F:arena" a release — and pins
  // the blocks an entry holds (position / block_tokens rounded up) so the
  // pool meters see them exactly as the real pool does.
  void set_prefix_arena(int slots, int64_t align, int64_t chunk_tokens = 2048) {
    arena_slots_ = slots;
    arena_align_ = align;
    arena_chunk_ = chunk_tokens;
  }
  dgpp::sched::SchedulerEngine::PrefixInfo prefix_info() const override {
    dgpp::sched::SchedulerEngine::PrefixInfo info;
    info.arena_slots = arena_slots_;
    info.align = arena_align_;
    info.block_tokens = block_tokens_;
    info.chunk_tokens = arena_chunk_;
    info.step_tokens_max = step_tokens_max_;
    info.prefill_lookahead = prefill_lookahead_;
    return info;
  }
  // The two-token step's hop (M7): the fake commits whatever its script
  // says per step; an armed hop is taken ("H:req:pos@slot") when the step
  // returns two tokens, at the position the scheduler named, which must be
  // the committed count + 1.
  void set_step_tokens_max(int n) { step_tokens_max_ = n; }
  void prefix_arm_hop(int req, int slot, int64_t position) override {
    require(live_.count(req) != 0, "fake: hop armed on an unopened slot");
    hop_armed_[req] = {slot, position};
  }
  int step_tokens_max_ = 1;
  bool prefill_lookahead_ = false;
  std::map<int, std::pair<int, int64_t>> hop_armed_;
  int32_t prefill_cached(int req, const std::vector<int64_t>& prompt,
                         dgpp::sched::SchedulerEngine::PrefixPrefill* plan) override {
    require(plan != nullptr && plan->boundaries != nullptr, "fake: plan");
    if (plan->attach_slot >= 0) {
      require(pinned_.count(plan->attach_slot) != 0, "fake: attach from an empty arena slot");
      require(pinned_positions_[plan->attach_slot] == plan->attach_position,
              "fake: attach position differs from the arena slot's");
      ops_.push_back("X:" + std::to_string(req) + ":" +
                     std::to_string(plan->attach_position) + "@" +
                     std::to_string(plan->attach_slot));
    }
    const int32_t token = prefill(req, prompt);
    if (plan->snap_slot >= 0) {
      require(pinned_.count(plan->snap_slot) == 0, "fake: snapshot into an occupied arena slot");
      pin(plan->snap_slot, plan->snap_position);
      plan->snap_taken = true;
      ops_.push_back("N:" + std::to_string(req) + ":" +
                     std::to_string(plan->snap_position) + "@" +
                     std::to_string(plan->snap_slot));
    }
    if (plan->body_snap_slot >= 0) {
      require(pinned_.count(plan->body_snap_slot) == 0, "fake: body snapshot slot occupied");
      pin(plan->body_snap_slot, plan->body_snap_position);
      plan->body_snap_taken = true;
      ops_.push_back("N:" + std::to_string(req) + ":" + std::to_string(plan->body_snap_position) +
                     "@" + std::to_string(plan->body_snap_slot));
    }
    return token;
  }
  void prefix_snapshot(int req, int slot, int64_t position) override {
    require(live_.count(req) != 0, "fake: rolling snapshot of an unopened slot");
    const Live& live = live_.at(req);
    // The scheduler's view of the committed position must be the fake's:
    // prompt + tokens returned so far - 1 (the last is pending).
    require(position == live.prompt_tokens + static_cast<int64_t>(live.returned) - 1,
            "fake: rolling snapshot position " + std::to_string(position) +
                " differs from the slot's committed " +
                std::to_string(live.prompt_tokens + static_cast<int64_t>(live.returned) - 1));
    if (pinned_.count(slot) != 0) pinned_.erase(slot);
    pin(slot, position);
    ops_.push_back("RS:" + std::to_string(req) + ":" + std::to_string(position) +
                   "@" + std::to_string(slot));
  }
  void prefix_release(int slot) override {
    // Like PrefixArena, releasing an armed but not yet taken snapshot is a no-op.
    if (pinned_.count(slot) == 0) return;
    pinned_.erase(slot);
    pinned_positions_.erase(slot);
    ops_.push_back("F:" + std::to_string(slot));
  }
  int64_t pinned_blocks() const {
    int64_t n = 0;
    for (const auto& [slot, blocks] : pinned_) n += blocks;
    return n;
  }

  // The NVMe cold tier (issue #26): an in-memory slab of `pages` records.
  // Spills and restores complete after `disk_latency_` polls (the tests
  // drive the commits through the scheduler like the journal does); the
  // fake records "DS:op:slot" (a spill begun), "DR:op:slot:pos" (a
  // restore begun) and pins the restored slot's blocks like a snapshot's.
  // Identities: every pinned slot carries a generation stamp, so a
  // restored entry's blocks are fresh identities, as the real pool's.
  void set_disk(int64_t pages, int64_t blob_pages = 1, int64_t min_tokens = 0, int latency = 1) {
    disk_pages_ = pages;
    disk_blob_pages_ = blob_pages;
    disk_min_tokens_ = min_tokens;
    disk_latency_ = latency;
  }
  void fail_next_disk_op() { disk_fail_next_ = true; }
  int64_t disk_ops_begun() const { return disk_ops_begun_; }
  dgpp::sched::SchedulerEngine::DiskInfo disk_info() const override {
    dgpp::sched::SchedulerEngine::DiskInfo i;
    i.enabled = disk_pages_ > 0;
    i.pages = disk_pages_;
    i.page_bytes = 4096;
    i.blob_pages = disk_blob_pages_;
    i.min_tokens = disk_min_tokens_;
    return i;
  }
  dgpp::sched::SchedulerEngine::DiskBlocks disk_entry_blocks(int slot) const override {
    require(pinned_.count(slot) != 0, "fake: disk blocks of an empty arena slot");
    dgpp::sched::SchedulerEngine::DiskBlocks b;
    const int64_t position = pinned_positions_.at(slot);
    const uint64_t gen = pinned_gen_.at(slot);
    for (int64_t i = 0; i < position / block_tokens_; ++i) b.identities.push_back((gen << 32) | static_cast<uint64_t>(i));
    b.partial_identity = position % block_tokens_ != 0 ? ((gen << 32) | 0xffffu) : 0;
    b.mtp_position = position;
    return b;
  }
  void disk_spill_begin(const dgpp::sched::SchedulerEngine::DiskSpill& sp) override {
    require(pinned_.count(sp.slot) != 0, "fake: spill of an empty arena slot");
    ops_.push_back("DS:" + std::to_string(sp.op) + ":" + std::to_string(sp.slot));
    disk_pending_.push_back({sp.op, disk_latency_, !disk_fail_next_});
    disk_fail_next_ = false;
    ++disk_ops_begun_;
  }
  dgpp::sched::SchedulerEngine::DiskBlocks disk_restore_begin(
      const dgpp::sched::SchedulerEngine::DiskRestore& rs) override {
    require(pinned_.count(rs.slot) == 0, "fake: restore into an occupied arena slot");
    const int64_t need = static_cast<int64_t>(rs.pages.size()) + (rs.partial_page >= 0 ? 1 : 0);
    require(pool_blocks_in_use() + need <= total_blocks_, "fake: restore without free blocks");
    // Every restored block is the entry's own (the real tier acquires
    // fresh blocks); the pin counts them all.
    pinned_[rs.slot] = need;
    pinned_positions_[rs.slot] = rs.position;
    pinned_gen_[rs.slot] = ++gen_;
    ops_.push_back("DR:" + std::to_string(rs.op) + ":" + std::to_string(rs.slot) + ":" +
                   std::to_string(rs.position));
    disk_pending_.push_back({rs.op, disk_latency_, !disk_fail_next_});
    disk_fail_next_ = false;
    ++disk_ops_begun_;
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
  struct PendingDisk {
    uint64_t op;
    int left;
    bool ok;
  };
  int64_t disk_pages_ = 0, disk_blob_pages_ = 1, disk_min_tokens_ = 0;
  int disk_latency_ = 1;
  bool disk_fail_next_ = false;
  int64_t disk_ops_begun_ = 0;
  std::vector<PendingDisk> disk_pending_;
  std::map<int, uint64_t> pinned_gen_;
  uint64_t gen_ = 0;

  int max_concurrent_requests() const override { return slots_; }
  int decode_batch_capacity() const override { return batch_capacity_; }
  int64_t pool_blocks_total() const override { return total_blocks_; }
  int64_t pool_blocks_in_use() const override {
    int64_t sum = 0;
    for (const auto& [slot, live] : live_) sum += live.held_blocks;
    return sum + pinned_blocks();
  }
  int64_t blocks_for_tokens(int64_t tokens) const override {
    return (tokens + block_tokens_ - 1) / block_tokens_;
  }

  int32_t prefill(int req, const std::vector<int64_t>& prompt) override {
    std::vector<Episode>& queue = episodes_[req];
    require(!queue.empty(), "fake: prefill on slot " + std::to_string(req) +
                               " with no armed episode");
    Live live;
    live.episode = std::move(queue.front());
    queue.erase(queue.begin());
    live.prompt_tokens = static_cast<int64_t>(prompt.size());
    live.last_token = live.episode.first;
    live.returned = 1;
    live_[req] = live;
    ops_.push_back("P:" + std::to_string(req) + ":" +
                   std::to_string(prompt.size()));
    note_logprobs(req, {live.last_token});
    return live.last_token;
  }
  void note_logprobs(int req, const std::vector<int32_t>& tokens) {
    if (report_.count(req) == 0 || report_[req] < 0) return;
    for (const int32_t t : tokens) {
      dgpp::sample::Result r;
      r.token = t;
      r.logprob = -static_cast<float>(t);
      r.top_logprobs.emplace_back(t, -static_cast<float>(t));
      pending_lps_[req].push_back(r);
    }
  }

  void reserve(int req, int64_t tokens) override {
    require(live_.count(req) != 0,
            "fake: reserve on unopened slot " + std::to_string(req));
    Live& live = live_[req];
    // Full-reserve pins the lifetime once; grow-on-demand (M6 6d) starts
    // inside it and grows monotonically, never past it.
    require(tokens >= live.prompt_tokens + 1 &&
                tokens <= live.prompt_tokens + live.episode.max_steps,
            "fake: reservation outside [prompt + 1, prompt + max_steps]");
    require(blocks_for_tokens(tokens) >= live.held_blocks,
            "fake: a reservation shrank");
    live.held_blocks = blocks_for_tokens(tokens);
  }

  std::vector<int32_t> step(int req) override {
    require(live_.count(req) != 0,
            "fake: step on unopened slot " + std::to_string(req));
    Live& live = live_[req];
    require(live.next_step < live.episode.steps.size(),
            "fake: token script exhausted for slot " +
                std::to_string(req) + " (a step the policy should not "
                "have issued — script the test correctly)");
    std::vector<int32_t> out = live.episode.steps[live.next_step++];
    const int32_t prev_token = live.last_token;
    if (!out.empty()) live.last_token = out.back();
    if (hop_armed_.count(req) != 0) {
      const auto [slot, position] = hop_armed_[req];
      hop_armed_.erase(req);
      if (out.size() >= 2) {
        require(position == live.prompt_tokens + static_cast<int64_t>(live.returned),
                "fake: hop position " + std::to_string(position) +
                    " is not the committed count + 1 (" +
                    std::to_string(live.prompt_tokens + static_cast<int64_t>(live.returned)) + ")");
        if (pinned_.count(slot) != 0) pinned_.erase(slot);
        pin(slot, position);
        ops_.push_back("H:" + std::to_string(req) + ":" + std::to_string(position) +
                       "@" + std::to_string(slot));
      }
    }
    live.returned += out.size();
    ops_.push_back("S:" + std::to_string(req) + ":" +
                   std::to_string(prev_token));
    note_logprobs(req, out);
    return out;
  }

  std::vector<std::vector<int32_t>> step_batch(
      const std::vector<int>& reqs) override {
    batch_calls_.push_back(reqs);
    return SchedulerEngine::step_batch(reqs);
  }

  void close(int req) override {
    require(live_.count(req) != 0,
            "fake: close on unopened slot " + std::to_string(req));
    require(live_[req].held_blocks > 0,
            "fake: close with no reservation on slot " +
                std::to_string(req) + " (double close?)");
    live_.erase(req);
    ops_.push_back("C:" + std::to_string(req));
  }

 private:
  struct Episode {
    int32_t first = -1;
    std::vector<std::vector<int32_t>> steps;
    int max_steps = 0;
  };
  struct Live {
    Episode episode;
    int64_t held_blocks = 0;
    int64_t prompt_tokens = 0;
    size_t next_step = 0;
    int32_t last_token = -1;
    size_t returned = 0;  // tokens handed to the scheduler (the pending one included)
  };
  void pin(int slot, int64_t position) {
    // Partial-pin mode models the real pool: an entry's full blocks are the
    // request's own (shared by reference), only its partial-block copy
    // costs a block, and the pool throws when overdrawn. The default counts
    // every block below the position (the older gates' arithmetic).
    pinned_[slot] = partial_pins_
                        ? (position % block_tokens_ != 0 ? 1 : 0)
                        : position / block_tokens_ + (position % block_tokens_ != 0 ? 1 : 0);
    pinned_positions_[slot] = position;
    pinned_gen_[slot] = ++gen_;
    if (partial_pins_ && pool_blocks_in_use() > total_blocks_)
      throw std::runtime_error("fake: pool overdrawn by a snapshot (" +
                               std::to_string(pool_blocks_in_use()) + " of " +
                               std::to_string(total_blocks_) + " blocks)");
  }

 public:
  void set_partial_pins(bool on) { partial_pins_ = on; }

 protected:
  bool partial_pins_ = false;

  int slots_;
  int arena_slots_ = 0;
  int64_t arena_align_ = 1;
  int64_t arena_chunk_ = 2048;
  std::map<int, int64_t> pinned_;           // arena slot -> blocks pinned
  std::map<int, int64_t> pinned_positions_;  // arena slot -> position
  int64_t total_blocks_;
  int64_t block_tokens_;
  int batch_capacity_ = 1;
  bool can_sample_ = false;
  std::map<int, int> report_;
  std::map<int, std::vector<dgpp::sample::Result>> pending_lps_;
  std::map<int, std::vector<Episode>> episodes_;
  std::map<int, Live> live_;
  std::vector<std::string> ops_;
  std::vector<std::vector<int>> batch_calls_;
};

SchedulerRequest make_request(const std::string& id, int prompt_len,
                              int max_steps, int cancel_after = 0) {
  SchedulerRequest r;
  r.id = id;
  r.prompt.assign(static_cast<size_t>(prompt_len), 7);
  r.max_steps = max_steps;
  r.cancel_after = cancel_after;
  return r;
}

// EOS id used by every script (token 999 in the streams).
constexpr int32_t kEos = 999;

std::string ids_joined(const std::vector<int64_t>& ids) {
  std::string s;
  for (int64_t t : ids) {
    if (!s.empty()) s += ",";
    s += std::to_string(t);
  }
  return s;
}

// A fake that takes prefill groups: cold prompts within `span` tokens
// admit together in one call (the op "PG:n" then the members' P ops).
class GroupFakeEngine : public FakeEngine {
 public:
  GroupFakeEngine(int slots, int64_t total_blocks, int64_t block_tokens, int64_t span, int64_t total)
      : FakeEngine(slots, total_blocks, block_tokens), span_(span), total_(total) {}
  int64_t prefill_group_span_limit() const override { return span_; }
  int64_t prefill_group_total_limit() const override { return total_; }
  std::vector<int32_t> prefill_group(const std::vector<int>& reqs,
                                     const std::vector<const std::vector<int64_t>*>& prompts) override {
    ops_.push_back("PG:" + std::to_string(reqs.size()));
    return SchedulerEngine::prefill_group(reqs, prompts);
  }
 private:
  int64_t span_, total_;
};

class ChunkFakeEngine : public FakeEngine {
 public:
  explicit ChunkFakeEngine(int64_t total_blocks = 100) : FakeEngine(2, total_blocks, 2, 2) {}
  int64_t prefill_chunk_alignment() const override { return 2; }
  int64_t prefill_chunk_limit() const override { return 16; }
  int64_t prefill_group_span_limit() const override { return 16; }
  int64_t prefill_group_total_limit() const override { return 32; }
  void begin_prefill(int req, const std::vector<int64_t>& prompt, int64_t reserved,
                     int64_t budget, const PrefixPrefill& plan) override {
    auto copy = plan;
    const int32_t first = plan.attach_slot >= 0 || plan.snap_slot >= 0 || plan.body_snap_slot >= 0
                              ? FakeEngine::prefill_cached(req, prompt, &copy)
                              : FakeEngine::prefill(req, prompt);
    FakeEngine::reserve(req, reserved);
    pending_[req] = {static_cast<int64_t>(prompt.size()) - plan.attach_position, budget, first,
                     copy.snap_taken, copy.body_snap_taken};
  }
  PrefillProgress advance_prefill(int req, int64_t budget = 0) override {
    auto& pending = pending_.at(req);
    const int64_t count = std::min(pending.remaining, budget > 0 ? budget : pending.budget);
    pending.remaining -= count;
    ops_.push_back("PF:" + std::to_string(req) + ":" + std::to_string(count));
    PrefillProgress out{count, pending.remaining == 0 ? pending.first : -1, pending.snap,
                        pending.body};
    if (pending.remaining == 0) pending_.erase(req);
    return out;
  }
  std::vector<int32_t> step(int req) override {
    require(!pending_.contains(req), "an unfinished prefill must not decode");
    return FakeEngine::step(req);
  }
  void close(int req) override {
    pending_.erase(req);
    FakeEngine::close(req);
  }
 private:
  struct Pending {
    int64_t remaining, budget;
    int32_t first;
    bool snap, body;
  };
  std::map<int, Pending> pending_;
};

DGPP_TEST(scheduler_serving_auto_prefill_budget_respects_engine_geometry_and_opt_out) {
  struct GeometryEngine : ChunkFakeEngine {
    int64_t align = 128, limit = 2048;
    int64_t prefill_chunk_alignment() const override { return align; }
    int64_t prefill_chunk_limit() const override { return limit; }
  } engine;
  dgpp::sched::AdmissionPolicy policy;
  policy.prefill_budget_tokens = -1;
  const auto resolved = [&] { return dgpp::serve::resolve_prefill_policy(policy, engine); };
  require(resolved().prefill_budget_tokens == 256, "automatic budget yields after 256 tokens");
  engine.limit = 128;
  require(resolved().prefill_budget_tokens == 128, "small forward limit caps the budget");
  engine.align = 512;
  engine.limit = 2048;
  require(resolved().prefill_budget_tokens == 512, "alignment wider than the default is honored");
  engine.align = 128;
  policy.prefill_idle_budget_tokens = 128;
  require(resolved().prefill_budget_tokens == 128, "explicit idle budget can bound the automatic busy budget");
  policy.prefill_idle_budget_tokens = 0;
  policy.prefill_budget_tokens = 0;
  require(resolved().prefill_budget_tokens == 0, "explicit full-prompt opt-out survives");
  policy.prefill_budget_tokens = 1024;
  require(resolved().prefill_budget_tokens == 1024, "explicit chunk size survives");
  policy.prefill_budget_tokens = -1;
  engine.align = engine.limit = 0;
  require(resolved().prefill_budget_tokens == 0, "unsupported engine keeps synchronous prefill");
}

DGPP_TEST(scheduler_chunked_prefill_bounds_work_and_keeps_decode_running) {
  ChunkFakeEngine engine;
  engine.arm(0, {10, 11, 12, 13, 14, 15, 16, 17}, 8);
  engine.arm(1, {20, 21, 22}, 3);
  dgpp::sched::AdmissionPolicy policy;
  policy.prefill_budget_tokens = 4;
  Scheduler sched(&engine, {}, 0, policy);
  SchedulerRequest decode;
  decode.id = "decode";
  decode.prompt = {1, 2};
  decode.max_steps = 8;
  sched.submit(decode);
  sched.tick();
  SchedulerRequest longer;
  longer.id = "long";
  longer.prompt.assign(11, 3);
  longer.max_steps = 3;
  sched.submit(longer);
  for (int tick = 0; tick < 2; ++tick) {
    const auto before = sched.meters().tokens_generated;
    sched.tick();
    require(sched.meters().tokens_generated == before + 1, "decode progresses between prefill chunks");
    require(sched.find("long")->steps_done == 0 && sched.meters().prefilling == 1,
            "partial chunks expose no generated tokens");
    require(sched.meters().prefilling == 1 && sched.meters().prompt_tokens_computed == 2 + 4 * (tick + 1),
            "in-progress prefill tokens are accounted once");
    require(sched.meters().prompt_tokens == sched.meters().prompt_tokens_computed,
            "logical and computed progress advance together; cold chunks are never negative cache hits");
    const auto progress = engine.prefill_monitor()->snapshot();
    require(progress.size() == 1 && progress[0].id == "long" && progress[0].total == 11 &&
                progress[0].processed == 4 * (tick + 1) && progress[0].cached == 0,
            "live monitor follows resumable chunk progress without reporting generated tokens");
  }
  sched.tick();
  require(sched.meters().tokens_generated > 5 && sched.meters().prefilling == 0,
          "only the final chunk publishes the first token");
  require(engine.prefill_monitor()->snapshot().empty(), "completed prefill leaves no live entry");
  sched.run_to_completion();
  require(sched.find("long")->generated == std::vector<int64_t>({20, 21, 22}), "chunked transcript");
  require(sched.meters().prompt_tokens_computed == 13 && sched.meters().pool_blocks_in_use == 0,
          "no double counting or reservation leak");
}

DGPP_TEST(scheduler_chunked_images_keep_decode_running_and_cancel_cleanly) {
  class ImageChunks : public ChunkFakeEngine {
   public:
    bool supports_images() const override { return true; }
    bool supports_image_prefix_cache() const override { return true; }
    bool supports_image_chunked_prefill() const override { return true; }
    void begin_prefill(int req, const std::vector<int64_t>& prompt, int64_t reserved,
                        int64_t budget, const PrefixPrefill& plan) override {
      require(plan.images && plan.images->size() == 12, "all historical images reach the cursor");
      require(plan.images->back().offset == 23 && plan.images->back().rgb[0] == 123,
              "image spans and pixels survive admission");
      ChunkFakeEngine::begin_prefill(req, prompt, reserved, budget, plan);
    }
  } engine;
  engine.arm(0, {10, 11, 12, 13, 14, 15, 16, 17}, 8);
  engine.arm(1, {20, 21, 22}, 3);
  dgpp::sched::AdmissionPolicy policy;
  policy.prefill_budget_tokens = 4;
  Scheduler sched(&engine, {}, 0, policy);
  sched.submit(make_request("decode", 2, 8));
  sched.tick();
  auto request = make_request("images", 25, 3);
  for (int i = 0; i < 12; ++i)
    request.images.push_back({2 * i + 1, 1, 28, 28, std::vector<uint8_t>(28 * 28 * 3, 123)});
  sched.submit(request);
  for (int i = 0; i < 3; ++i) {
    const auto tokens = sched.meters().tokens_generated;
    sched.tick();
    require(sched.meters().tokens_generated == tokens + 1, "decode continues during image prefill");
    require(sched.find("images")->steps_done == 0, "unfinished images cannot decode");
  }
  require(sched.cancel("images"), "image prefill can be cancelled at a yield");
  sched.tick();
  require(sched.meters().prefilling == 0 && engine.prefill_monitor()->snapshot().empty(),
          "cancelled image cursor released");
  while (sched.tick()) {}
  require(sched.meters().pool_blocks_in_use == 0, "image cancellation releases reservation");
}

DGPP_TEST(scheduler_chunked_prefill_yields_reservation_to_older_decode) {
  ChunkFakeEngine engine(10);
  engine.arm(0, {10, 11, 12, 13, 14, 15, 16, 17}, 8);
  engine.arm(1, {20, 21, 22}, 3);
  dgpp::sched::AdmissionPolicy policy;
  policy.mode = dgpp::sched::AdmissionPolicy::Mode::kGrowOnDemand;
  policy.window_tokens = 2;
  policy.prefill_budget_tokens = 4;
  Scheduler sched(&engine, {}, 0, policy);
  sched.submit(make_request("older", 2, 8));
  sched.tick();
  sched.submit(make_request("younger", 11, 3));
  sched.tick();
  sched.tick();
  require(sched.meters().prefilling == 1 && sched.meters().pool_blocks_in_use == 10,
          "the unfinished younger request owns the remaining reservation");
  sched.tick();
  const auto* younger = sched.find("younger");
  require(younger->reason == Scheduler::Result::Reason::kPoolExhausted && younger->generated.empty(),
          "shed unfinished younger prefill before interrupting older decode");
  sched.run_to_completion();
  require(sched.find("older")->generated == std::vector<int64_t>({10, 11, 12, 13, 14, 15, 16, 17}),
          "older transcript survives pool pressure");
  require(sched.meters().prompt_tokens_computed == 10 && sched.meters().prompt_tokens == 10 &&
              sched.meters().pool_blocks_in_use == 0,
          "partial work is counted and the reservation is released");
}

DGPP_TEST(scheduler_chunked_prefill_cancel_at_each_yield_releases_cache_and_reservations) {
  for (const int chunks : {1, 2}) {
    ChunkFakeEngine engine;
    engine.set_prefix_arena(4, 2, 4);
    engine.arm(0, {10, 11}, 2);
    engine.arm(0, {20, 21}, 2);
    dgpp::sched::AdmissionPolicy policy;
    policy.prefill_budget_tokens = 4;
    Scheduler sched(&engine, {}, 0, policy);
    sched.set_keep_retired(false);
    SchedulerRequest request;
    request.id = "cancel";
    request.prompt.assign(11, 3);
    request.max_steps = 2;
    request.boundaries = {4, 8};
    sched.submit(request);
    for (int i = 0; i < chunks; ++i) sched.tick();
    require(sched.meters().prefix_entries == 0, "an unfinished prefill does not publish a cache entry");
    require(sched.cancel("cancel"), "in-progress prefill is cancellable");
    sched.tick();
    require(engine.prefill_monitor()->snapshot().empty(), "cancelled prefill leaves no live entry");
    require(!sched.has_pending() && sched.meters().pool_blocks_in_use == 0 && sched.meters().records == 0,
            "cancellation releases snapshot, blocks and compacted request record");
    request.id = "reuse";
    request.prompt = {4, 5};
    request.boundaries.clear();
    sched.submit(request);
    sched.run_to_completion();
    require(sched.meters().active == 0 &&
                sched.meters().pool_blocks_in_use == sched.meters().prefix_blocks_pinned,
            "the reused slot releases its reservation; only completed cache entries remain");
  }
}

DGPP_TEST(scheduler_chunked_prefill_group_respects_the_total_tick_budget) {
  for (const int first_length : {3, 5}) {
    ChunkFakeEngine engine;
    engine.arm(0, {10, 11}, 2);
    engine.arm(0, {20, 21}, 2);  // a deferred second prompt reuses the lowest free slot
    engine.arm(1, {20, 21}, 2);
    dgpp::sched::AdmissionPolicy policy;
    policy.prefill_budget_tokens = 8;
    Scheduler sched(&engine, {}, 0, policy);
    sched.submit(make_request("first", first_length, 2));
    sched.submit(make_request("second", 5, 2));
    sched.tick();
    require(sched.meters().prompts_prefilled == (first_length == 3 ? 2 : 1),
            "a group contains only the prompts that fit the tick's token budget");
    sched.run_to_completion();
  }
}

DGPP_TEST(scheduler_idle_prefill_budget_tracks_decode_retirement) {
  ChunkFakeEngine engine;
  engine.arm(0, {10, 11, 12}, 3);
  engine.arm(1, {20, 21}, 2);
  dgpp::sched::AdmissionPolicy policy;
  policy.prefill_budget_tokens = 4;
  policy.prefill_idle_budget_tokens = 16;
  Scheduler sched(&engine, {}, 0, policy);
  sched.submit(make_request("peer", 2, 3));
  sched.tick();
  sched.submit(make_request("long", 23, 2));
  sched.tick();
  require(sched.find("peer")->status == Scheduler::Result::Status::kDone &&
              sched.meters().prompt_tokens_computed == 6,
          "use the busy budget for the tick in which the decoding peer finishes");
  sched.tick();
  require(sched.meters().prompt_tokens_computed == 22 && sched.meters().prefilling == 1,
          "increase the next chunk's budget after the peer retires");
  sched.run_to_completion();
  require(sched.find("long")->generated == std::vector<int64_t>({20, 21}) &&
              sched.meters().prompt_tokens_computed == 25 && sched.meters().pool_blocks_in_use == 0,
          "resizing preserves completion, accounting and reservation release");
}

DGPP_TEST(scheduler_idle_prefill_budget_applies_to_first_chunk_and_groups) {
  for (const bool grouped : {false, true}) {
    ChunkFakeEngine engine;
    engine.arm(0, {10, 11}, 2);
    engine.arm(1, {20, 21}, 2);
    dgpp::sched::AdmissionPolicy policy;
    policy.prefill_budget_tokens = 4;
    policy.prefill_idle_budget_tokens = 16;
    Scheduler sched(&engine, {}, 0, policy);
    sched.submit(make_request("first", grouped ? 5 : 23, 2));
    if (grouped) sched.submit(make_request("second", 5, 2));
    sched.tick();
    require(sched.meters().prompt_tokens_computed == (grouped ? 10 : 16),
            "idle admission uses the larger budget for both groups and continuations");
    if (!grouped) {
      require(sched.cancel("first"), "idle continuation can be cancelled at its first yield");
      sched.tick();
    }
    sched.run_to_completion();
    require(sched.meters().pool_blocks_in_use == 0, "idle admission releases reservations");
  }
}

DGPP_TEST(scheduler_chunked_prefill_budget_rejects_unsupported_and_unaligned_configs) {
  FakeEngine old(2, 100, 2);
  ChunkFakeEngine chunked;
  for (auto* engine : std::vector<SchedulerEngine*>{&old, &chunked}) {
    dgpp::sched::AdmissionPolicy policy;
    policy.prefill_budget_tokens = 3;
    bool rejected = false;
    try { Scheduler sched(engine, {}, 0, policy); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "invalid continuation policy must fail before admission");
  }
  for (const auto& [busy, idle] : std::vector<std::pair<int, int>>{{0, 16}, {4, -2}, {4, 2}, {4, 7}, {4, 18}}) {
    dgpp::sched::AdmissionPolicy policy;
    policy.prefill_budget_tokens = busy;
    policy.prefill_idle_budget_tokens = idle;
    bool rejected = false;
    try { Scheduler sched(&chunked, {}, 0, policy); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "idle budget requires enabled, ordered, aligned budgets within scratch capacity");
  }
}

DGPP_TEST(scheduler_groupPrefill_admitsQueuedShortPromptsTogether) {
  // GIVEN three 5-token requests queued at once, 3 slots, an engine that
  // takes groups of prompts within 8 tokens: one tick admits all three in
  // one prefill (PG:3 then each slot's P), then the steps round-robin.
  GroupFakeEngine engine(/*slots=*/3, /*total_blocks=*/100, /*block_tokens=*/4, /*span=*/8, /*total=*/64);
  engine.arm(0, {1, 2}, /*max_steps=*/2);
  engine.arm(1, {4, 5}, /*max_steps=*/2);
  engine.arm(2, {7, 8}, /*max_steps=*/2);
  Scheduler sched(&engine, {kEos});
  sched.submit(make_request("a", 5, 2));
  sched.submit(make_request("b", 5, 2));
  sched.submit(make_request("c", 5, 2));
  try {
    sched.run_to_completion();
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string(e.what()) + "\n  ops so far: " + engine.op_stream());
  }
  const std::string got = engine.op_stream();
  require(got.rfind("PG:3 P:0:5 P:1:5 P:2:5", 0) == 0,
          "the three queued prompts admit as one group:\n  got: " + got);
  require(sched.results().size() == 3, "three results");
  const auto meters = sched.meters();
  require(std::abs(meters.prefill_request_ms - 3 * meters.prefill_ms) < 1e-6,
          "a grouped prefill counts execution once and request wait per member");
  require(ids_joined(sched.results()[0].generated) == "1,2", "a ids");
  require(ids_joined(sched.results()[1].generated) == "4,5", "b ids");
  require(ids_joined(sched.results()[2].generated) == "7,8", "c ids");
  // A prompt past the span limit admits alone (the group stops at it).
  GroupFakeEngine engine2(/*slots=*/3, /*total_blocks=*/100, /*block_tokens=*/4, /*span=*/8, /*total=*/64);
  engine2.arm(0, {1}, 1);  // a
  engine2.arm(1, {4}, 1);  // c pairs with a
  engine2.arm(0, {7}, 1);  // b admits alone after a retires, in the lowest free slot
  Scheduler sched2(&engine2, {kEos});
  sched2.submit(make_request("a", 5, 1));
  sched2.submit(make_request("b", 9, 1));  // past the span limit
  sched2.submit(make_request("c", 5, 1));
  try {
    sched2.run_to_completion();
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string(e.what()) + "\n  ops so far: " + engine2.op_stream());
  }
  const std::string got2 = engine2.op_stream();
  require(got2.rfind("PG:2 P:0:5 P:1:5", 0) == 0,
          "the long prompt stays out of the group; the short ones pair:\n  got: " + got2);
  require(got2.find("P:0:9") != std::string::npos, "the long prompt admits alone:\n  got: " + got2);
}

DGPP_TEST(scheduler_strictAlternation_pinnedOpSequence) {
  // GIVEN three 5-token requests (max 3 steps each), 2 slots, an ample
  // pool; slots arm in the deterministic assignment order (a->0, b->1,
  // c reuses 0 after a retires):
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.arm(0, {1, 2, 3}, /*max_steps=*/3);   // a
  engine.arm(1, {4, 5, 6}, /*max_steps=*/3);   // b
  engine.arm(0, {7, 8, 9}, /*max_steps=*/3);   // c (slot 0's 2nd episode)
  Scheduler sched(&engine, {kEos});
  sched.submit(make_request("a", 5, 3));
  sched.submit(make_request("b", 5, 3));
  sched.submit(make_request("c", 5, 3));

  // WHEN the scheduler runs to completion,
  sched.run_to_completion();

  // THEN the op stream is exactly the pinned policy: one admission per
  // tick before the step, round-robin by arrival, retirement-triggered
  // admission of c, lowest-free-slot reuse, and close on every retire.
  const std::string expected =
      "P:0:5 S:0:1 P:1:5 S:1:4 S:0:2 C:0 P:0:5 S:1:5 C:1 S:0:7 S:0:8 C:0";
  require(engine.op_stream() == expected,
          "op stream drifted from the pinned policy:\n  got:      " +
              engine.op_stream() + "\n  expected: " + expected);
  require(sched.results().size() == 3, "three results");
  for (const auto& res : sched.results()) {
    require(res.status == Scheduler::Result::Status::kDone,
            "all three retire Done (steps cap)");
    require(res.reason == Scheduler::Result::Reason::kSteps,
            "all three retire on the steps cap");
    require(res.steps_done == 3, "each generated 3 tokens");
  }
  require(ids_joined(sched.results()[0].generated) == "1,2,3", "a ids");
  require(ids_joined(sched.results()[1].generated) == "4,5,6", "b ids");
  require(ids_joined(sched.results()[2].generated) == "7,8,9", "c ids");
}

DGPP_TEST(scheduler_batchEngine_stepsRoundRobinSliceInOnePass) {
  // GIVEN a two-row engine and two requests. Strict admission still opens
  // at most one request per tick; once both are active, one physical pass
  // receives both slots in round-robin order.
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/100, /*block_tokens=*/4,
                    /*batch_capacity=*/2);
  engine.arm(0, {10, 11, 12}, /*max_steps=*/3);  // a
  engine.arm(1, {20, 21, 22}, /*max_steps=*/3);  // b
  Scheduler sched(&engine, {kEos});
  sched.submit(make_request("a", 5, 3));
  sched.submit(make_request("b", 5, 3));

  sched.run_to_completion();

  // Tick 1 carries a. Tick 2 admits b, then rotates from a to [b,a] in one
  // call (a retires independently at its cap). Tick 3 carries b alone.
  const std::vector<std::vector<int>> want_batches = {{0}, {1, 0}, {1}};
  require(engine.batch_calls() == want_batches,
          "batch engine did not receive the canonical round-robin slices");
  require(ids_joined(sched.results()[0].generated) == "10,11,12",
          "batched a transcript");
  require(ids_joined(sched.results()[1].generated) == "20,21,22",
          "batched b transcript");
}

DGPP_TEST(scheduler_meters_countPromptsStepsAndRowsForTheThroughputLine) {
  // GIVEN the slice test's two-row engine and two requests (5- and 7-token
  // prompts, three tokens each),
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/100, /*block_tokens=*/4,
                    /*batch_capacity=*/2);
  engine.arm(0, {10, 11, 12}, /*max_steps=*/3);  // a
  engine.arm(1, {20, 21, 22}, /*max_steps=*/3);  // b
  Scheduler sched(&engine, {kEos});
  sched.submit(make_request("a", 5, 3));
  sched.submit(make_request("b", 7, 3));
  require(sched.meters().prompts_prefilled == 0 &&
              sched.meters().decode_steps == 0,
          "nothing counted before the first tick");

  sched.run_to_completion();

  // THEN the throughput line's counters read the passes the slice test
  // pinned: two prefills of 12 prompt tokens, all computed (no cache);
  // three decode passes carrying four request-rows ({a}, {b,a}, {b}); six
  // tokens; timings that are at least not negative on a fake.
  const Scheduler::Meters m = sched.meters();
  require(m.prompts_prefilled == 2 && m.prompt_tokens == 12 &&
              m.prompt_tokens_computed == 12,
          "two prompts, 12 tokens, none from a cache");
  require(m.decode_steps == 3 && m.decode_rows == 4,
          "three decode passes over four request-rows");
  require(m.tokens_generated == 6, "six tokens generated");
  require(m.prefill_ms >= 0.0 && m.step_ms >= 0.0,
          "the engine timings are non-negative");
}

DGPP_TEST(scheduler_batchEngine_capacityBelowActive_rotatesFairly) {
  // GIVEN a three-slot engine whose physical pass carries only two rows.
  // Once three requests are active, each tick advances the next two in
  // arrival rotation and the cursor lands on the slice's last member, so
  // every request is stepped in two of every three ticks.
  FakeEngine engine(/*slots=*/3, /*total_blocks=*/100, /*block_tokens=*/4,
                    /*batch_capacity=*/2);
  engine.arm(0, {10, 11, 12, 13, 14}, /*max_steps=*/5);  // a
  engine.arm(1, {20, 21, 22, 23, 24}, /*max_steps=*/5);  // b
  engine.arm(2, {30, 31, 32, 33, 34}, /*max_steps=*/5);  // c
  Scheduler sched(&engine, {kEos});
  sched.submit(make_request("a", 5, 5));
  sched.submit(make_request("b", 5, 5));
  sched.submit(make_request("c", 5, 5));

  sched.run_to_completion();

  // Tick 1: admit a, step {a}. Tick 2: admit b, rotate from a: {b, a}.
  // Tick 3: admit c, rotate from a: {b, c}. Tick 4: from c: {a, b}.
  // Tick 5: from b: {c, a} — a reaches its cap here. Tick 6: {b, c}; b
  // retires. Tick 7: {c}; c retires.
  const std::vector<std::vector<int>> want_batches = {
      {0}, {1, 0}, {1, 2}, {0, 1}, {2, 0}, {1, 2}, {2}};
  require(engine.batch_calls() == want_batches,
          "capacity-2 slices did not rotate fairly over three requests");
  require(ids_joined(sched.results()[0].generated) == "10,11,12,13,14",
          "rotated a transcript");
  require(ids_joined(sched.results()[1].generated) == "20,21,22,23,24",
          "rotated b transcript");
  require(ids_joined(sched.results()[2].generated) == "30,31,32,33,34",
          "rotated c transcript");
}

DGPP_TEST(scheduler_poolBudget_defersAdmissionUntilPeerRetires) {
  // GIVEN two requests each reserving 2 blocks (5+3 tokens at 4
  // tokens/block) against a 2-block pool — only one fits at a time:
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/2, /*block_tokens=*/4);
  engine.arm(0, {1, 2, 3}, /*max_steps=*/3);  // a
  engine.arm(0, {4, 5, 6}, /*max_steps=*/3);  // b (admits after a frees)
  Scheduler sched(&engine, {kEos});
  sched.submit(make_request("a", 5, 3));
  sched.submit(make_request("b", 5, 3));

  // WHEN run to completion,
  sched.run_to_completion();

  // THEN b is deferred (no P until a's C), admitted on the very next
  // tick, and both finish normally — the budget, not the slot count,
  // gated b (2 slots were open the whole time).
  const std::string expected =
      "P:0:5 S:0:1 S:0:2 C:0 P:0:5 S:0:4 S:0:5 C:0";
  require(engine.op_stream() == expected,
          "budget-deferral stream drifted:\n  got:      " +
              engine.op_stream() + "\n  expected: " + expected);
  require(sched.results()[0].status == Scheduler::Result::Status::kDone,
          "a Done");
  require(sched.results()[1].status == Scheduler::Result::Status::kDone,
          "b Done");
}

DGPP_TEST(scheduler_capacityBelowSmallestReservation_throwsDeadlock) {
  // GIVEN one request needing 2 blocks against a 1-block pool — it can
  // never fit and nothing else can retire to free space:
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/1, /*block_tokens=*/4);
  Scheduler sched(&engine, {kEos});
  sched.submit(make_request("a", 5, 3));

  // WHEN run, THEN the scheduler rejects the input with an error (an operator sizing
  // error — never a hang, never a silent partial run).
  bool threw = false;
  try {
    sched.run_to_completion();
  } catch (const std::runtime_error& e) {
    threw = true;
    require(std::string(e.what()).find("admission deadlock") !=
                std::string::npos,
            "the error must name the deadlock: " + std::string(e.what()));
  }
  require(threw, "run_to_completion must throw on an unfixable queue");
  require(engine.ops().empty(), "no ops may run before the deadlock");
}

DGPP_TEST(scheduler_eosOnPrefillPick_retiresImmediatelyThenPeerAdmits) {
  // GIVEN a whose first picked token is EOS (the model answered in one
  // token) and b behind it:
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.arm(0, {kEos}, /*max_steps=*/5);    // a: EOS on the prefill pick
  engine.arm(0, {7, 8}, /*max_steps=*/2);    // b: reuses the freed slot
  Scheduler sched(&engine, {kEos});
  sched.submit(make_request("a", 5, 5));
  sched.submit(make_request("b", 5, 2));

  // WHEN run,
  sched.run_to_completion();

  // THEN a retires with a single generated token (Done/eos — no step
  // op for it), b admits on the next tick, and b's steps cap ends it.
  const std::string expected = "P:0:5 C:0 P:0:5 S:0:7 C:0";
  require(engine.op_stream() == expected,
          "prefill-EOS stream drifted:\n  got:      " + engine.op_stream() +
              "\n  expected: " + expected);
  const auto& a = sched.results()[0];
  require(a.status == Scheduler::Result::Status::kDone &&
             a.reason == Scheduler::Result::Reason::kEos &&
             a.steps_done == 1 && ids_joined(a.generated) == "999",
          "a: one token, Done/eos");
  const auto& b = sched.results()[1];
  require(b.status == Scheduler::Result::Status::kDone &&
             b.reason == Scheduler::Result::Reason::kSteps &&
             ids_joined(b.generated) == "7,8",
          "b: steps cap, both tokens");
}

DGPP_TEST(scheduler_cancellation_isolatedPeerTranscriptUnchanged) {
  // GIVEN the headline property: x runs twice — once solo, once
  // interleaved with y, which is cancelled after 2 of its 4 tokens.
  const auto run_pair = [&](bool with_y) {
    FakeEngine engine(/*slots=*/2, /*total_blocks=*/100, /*block_tokens=*/4);
    engine.arm(0, {1, 2, 3, 4}, /*max_steps=*/4);  // x -> slot 0
    if (with_y) engine.arm(1, {5, 6, 7, 8}, /*max_steps=*/4);  // y
    Scheduler sched(&engine, {kEos});
    sched.submit(make_request("x", 5, 4));
    if (with_y) sched.submit(make_request("y", 5, 4, /*cancel_after=*/2));
    sched.run_to_completion();
    return std::make_pair(engine.op_stream(), sched.results()[0]);
  };

  // WHEN both runs complete,
  const auto [solo_stream, solo_x] = run_pair(false);
  const auto [pair_stream, pair_x] = run_pair(true);

  // THEN x's transcript is IDENTICAL alone vs interleaved-with-a-
  // cancelled-peer: same ids, same status, same step count. The peer's
  // cancellation released its slot mid-flight and touched nothing else.
  require(ids_joined(solo_x.generated) == "1,2,3,4", "x solo ids");
  require(ids_joined(pair_x.generated) == "1,2,3,4",
          "x interleaved ids must equal solo: " +
              ids_joined(pair_x.generated));
  require(solo_x.status == pair_x.status &&
             solo_x.reason == pair_x.reason &&
             solo_x.steps_done == pair_x.steps_done,
          "x's terminal state must be identical across runs");
  // And the pair run shows y admitted, stepped twice, cancelled, closed
  // — while x keeps decoding to its cap.
  const std::string expected =
      "P:0:5 S:0:1 P:1:5 S:1:5 C:1 S:0:2 S:0:3 C:0";
  require(pair_stream == expected,
          "pair stream drifted:\n  got:      " + pair_stream +
              "\n  expected: " + expected);
}

DGPP_TEST(scheduler_sameScenarioTwice_identicalOpStreams) {
  // GIVEN a scenario with deferral, cancellation, and EOS mixed,
  const auto run = [&]() {
    FakeEngine engine(/*slots=*/2, /*total_blocks=*/3, /*block_tokens=*/4);
    // Pool fits one reservation at a time; x EOSes on its 3rd token
    // (before its cap), freeing space for y early.
    engine.arm(0, {1, 2, kEos}, /*max_steps=*/4);  // x -> slot 0
    engine.arm(0, {7, 8}, /*max_steps=*/2);        // y -> reuses slot 0
    Scheduler sched(&engine, {kEos});
    sched.submit(make_request("x", 5, 4));
    sched.submit(make_request("y", 5, 2, /*cancel_after=*/1));
    sched.run_to_completion();
    std::vector<std::string> summary;
    for (const auto& r : sched.results())
      summary.push_back(std::to_string(static_cast<int>(r.status)) + ":" +
                        ids_joined(r.generated));
    return std::make_pair(engine.op_stream(), summary);
  };

  // WHEN run twice,
  const auto first = run();
  const auto second = run();

  // THEN the op stream AND every result are identical — the §11
  // identical-rank-order invariant in miniature (if two runs of the
  // same pure function ever disagree, the fabric would desync).
  require(first.first == second.first,
          "op streams must be identical across runs");
  require(first.second == second.second,
          "results must be identical across runs");
}

DGPP_TEST(scheduler_eosMidRun_retiresBeforeStepsCap) {
  // GIVEN a request whose 3rd token is EOS with a 5-step cap,
  FakeEngine engine(/*slots=*/1, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.arm(0, {1, 2, kEos}, /*max_steps=*/5);
  Scheduler sched(&engine, {kEos});
  sched.submit(make_request("a", 5, 5));

  // WHEN run,
  sched.run_to_completion();

  // THEN it retires Done/eos at 3 tokens — EOS dominates the cap.
  const std::string expected = "P:0:5 S:0:1 S:0:2 C:0";
  require(engine.op_stream() == expected,
          "mid-run EOS stream drifted:\n  got:      " + engine.op_stream() +
              "\n  expected: " + expected);
  const auto& a = sched.results()[0];
  require(a.reason == Scheduler::Result::Reason::kEos && a.steps_done == 3,
          "Done/eos at 3 tokens");
}

DGPP_TEST(scheduler_oneTokenCap_retiresOnPrefillWithoutDecode) {
  // GIVEN a request whose whole output budget is the prefill pick,
  FakeEngine engine(/*slots=*/1, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.arm(0, {17}, /*max_steps=*/1);
  Scheduler sched(&engine, {kEos});
  sched.submit(make_request("one", 5, 1));

  // WHEN it runs, THEN no decode step is issued: the prefill pick is token
  // one by the SchedulerRequest contract and therefore meets the cap.
  sched.run_to_completion();
  require(engine.op_stream() == "P:0:5 C:0",
          "max_steps=1 must not issue an extra decode: " +
              engine.op_stream());
  const auto& out = sched.results()[0];
  require(out.reason == Scheduler::Result::Reason::kSteps &&
              out.steps_done == 1 && ids_joined(out.generated) == "17",
          "one-token cap must return exactly the prefill pick");
}

DGPP_TEST(scheduler_multiTokenStep_eosInMiddleDropsSuffix) {
  // GIVEN one speculative pass that returns a token, EOS, then a token the
  // device happened to decide after the public request had ended,
  FakeEngine engine(/*slots=*/1, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.arm_batches(0, /*first=*/1, {{2, kEos, 77}}, /*max_steps=*/8);
  Scheduler sched(&engine, {kEos});
  sched.submit(make_request("spec", 5, 8));

  // WHEN it runs, THEN EOS is observable and the suffix is not.
  sched.run_to_completion();
  require(engine.op_stream() == "P:0:5 S:0:1 C:0",
          "one speculative pass then close");
  const auto& out = sched.results()[0];
  require(out.reason == Scheduler::Result::Reason::kEos &&
              out.steps_done == 3 && ids_joined(out.generated) == "1,2,999",
          "tokens after an in-batch EOS must be dropped");
}

DGPP_TEST(scheduler_multiTokenStep_capOvershootDropsSuffix) {
  // GIVEN two output slots left and a speculative pass returning three,
  FakeEngine engine(/*slots=*/1, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.arm_batches(0, /*first=*/1, {{2, 3, 4}}, /*max_steps=*/3);
  Scheduler sched(&engine, {kEos});
  sched.submit(make_request("spec", 5, 3));

  // WHEN it runs, THEN exactly the remaining two land in the transcript.
  sched.run_to_completion();
  const auto& out = sched.results()[0];
  require(out.reason == Scheduler::Result::Reason::kSteps &&
              out.steps_done == 3 && ids_joined(out.generated) == "1,2,3",
          "a speculative batch must truncate exactly at max_steps");
}

using dgpp::sched::QueueFullError;
using dgpp::sched::SchedulerObserver;

// The recording observer: every event in global fire order (the SSE tap
// in miniature — ordering IS the contract being tested).
class RecordingObserver : public SchedulerObserver {
 public:
  struct Event {
    std::string kind;  // "tok" | "end"
    std::string id;
    int64_t token = -1;
    int steps_done = -1;
    Scheduler::Result::Status status{};
  };
  void on_token(const std::string& id, int64_t token,
                int steps_done) override {
    events_.push_back({"tok", id, token, steps_done, {}});
  }
  void on_retire(const std::string& id,
                 const Scheduler::Result& result) override {
    events_.push_back({"end", id, -1, result.steps_done, result.status});
  }
  const std::vector<Event>& events() const { return events_; }
  std::string replay() const {
    std::string s;
    for (const Event& e : events_) {
      if (!s.empty()) s += " ";
      if (e.kind == "tok") s += "T(" + e.id + "," + std::to_string(e.token) + ")";
      else s += "E(" + e.id + "," + std::to_string(e.steps_done) + ")";
    }
    return s;
  }

 private:
  std::vector<Event> events_;
};

// DGPP_TEST(scheduler_observer...) uses this; keep helpers below the
// observer so the replay formatting stays next to the contract.
size_t count_kind(const RecordingObserver& o, const std::string& kind,
                  const std::string& id) {
  size_t n = 0;
  for (const auto& e : o.events())
    if (e.kind == kind && e.id == id) ++n;
  return n;
}

DGPP_TEST(scheduler_submit_rejectsManifestErrorsLoudly) {
  // GIVEN a scheduler with one request,
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/100, /*block_tokens=*/4);
  Scheduler sched(&engine, {kEos});
  sched.submit(make_request("a", 5, 3));

  // WHEN malformed requests are submitted, THEN each refusal names the
  // offense (identical on every rank — a partial manifest would
  // deadlock the fabric).
  const auto throws_with = [&](SchedulerRequest r, const std::string& needle) {
    try {
      sched.submit(std::move(r));
    } catch (const std::invalid_argument& e) {
      require(std::string(e.what()).find(needle) != std::string::npos,
              "refusal must name it: " + std::string(e.what()));
      return;
    }
    throw std::runtime_error("submit should have refused: " + needle);
  };
  throws_with(make_request("a", 5, 3), "duplicate");
  throws_with(make_request("b", 0, 3), "empty prompt");
  throws_with(make_request("b", 5, 0), "at least one token");
  throws_with(make_request("b", 5, 3, /*cancel_after=*/4), "cancel_after");
}

// ---------------------------------------------------------------------------
// Stage 4 surface: tick mode, dynamic arrival, bounded queue, external
// cancel, the observer, meters.
// ---------------------------------------------------------------------------

DGPP_TEST(scheduler_tickLoop_matchesRunToCompletion_exactly) {
  // GIVEN a scenario mixing deferral, scripted cancellation, and EOS,
  const auto run = [&](bool tick_mode) {
    FakeEngine engine(/*slots=*/2, /*total_blocks=*/3, /*block_tokens=*/4);
    engine.arm(0, {1, 2, kEos}, /*max_steps=*/4);  // x -> slot 0
    engine.arm(0, {7, 8}, /*max_steps=*/2);        // y -> reuses slot 0
    Scheduler sched(&engine, {kEos});
    sched.submit(make_request("x", 5, 4));
    sched.submit(make_request("y", 5, 2, /*cancel_after=*/1));
    // WHEN driven either by run_to_completion or by raw ticks,
    if (tick_mode)
      while (sched.tick()) {
      }
    else
      sched.run_to_completion();
    std::vector<std::string> summary;
    for (const auto& r : sched.results())
      summary.push_back(std::to_string(static_cast<int>(r.status)) + ":" +
                        ids_joined(r.generated));
    return std::make_pair(engine.op_stream(), summary);
  };

  // THEN both drivers produce the IDENTICAL op stream and results — the
  // tick extraction must be a pure refactor (the fabric's manifest path
  // rides on it staying bit-equal).
  const auto batch = run(false);
  const auto ticked = run(true);
  require(batch.first == ticked.first,
          "tick-mode op stream drifted from run_to_completion:\n  batch: " +
              batch.first + "\n  tick:  " + ticked.first);
  require(batch.second == ticked.second,
          "tick-mode results drifted from run_to_completion");
}

DGPP_TEST(scheduler_dynamicArrival_transcriptsMatchSolo) {
  // GIVEN a generating mid-answer, with b arriving mid-run,
  const auto solo = [&](const char* id, std::vector<int32_t> script,
                        int max_steps) {
    FakeEngine engine(/*slots=*/2, /*total_blocks=*/100, /*block_tokens=*/4);
    engine.arm(0, std::move(script), max_steps);
    Scheduler sched(&engine, {kEos});
    sched.submit(make_request(id, 5, max_steps));
    sched.run_to_completion();
    return ids_joined(sched.results()[0].generated);
  };
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.arm(0, {1, 2, 3, 4}, /*max_steps=*/4);  // a -> slot 0
  engine.arm(1, {5, 6}, /*max_steps=*/2);       // b -> slot 1
  Scheduler sched(&engine, {kEos});
  sched.submit(make_request("a", 5, 4));

  // WHEN a ticks twice alone, b submits mid-run, and the loop drives to
  // completion (the service's exact arrival shape),
  require(sched.tick(), "tick 1 works");
  require(sched.tick(), "tick 2 works");
  sched.submit(make_request("b", 5, 2));
  while (sched.tick()) {
  }

  // THEN the op stream shows b admitted at the next tick's alternation
  // slot, and both transcripts equal their solo runs — dynamic arrival
  // preserves the isolation property.
  const std::string expected =
      "P:0:5 S:0:1 S:0:2 P:1:5 S:1:5 C:1 S:0:3 C:0";
  require(engine.op_stream() == expected,
          "dynamic-arrival stream drifted:\n  got:      " +
              engine.op_stream() + "\n  expected: " + expected);
  require(ids_joined(sched.results()[0].generated) ==
              solo("a", {1, 2, 3, 4}, 4),
          "a's transcript must match its solo run");
  require(ids_joined(sched.results()[1].generated) == solo("b", {5, 6}, 2),
          "b's transcript must match its solo run");
  require(!sched.has_pending(), "nothing pending after completion");
}

DGPP_TEST(scheduler_boundedQueue_shedsWhenFullThenAdmits) {
  // GIVEN a scheduler with a one-deep admission queue,
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.arm(0, {1, 2, 3}, /*max_steps=*/3);  // a -> slot 0
  engine.arm(1, {4, 5, 6}, /*max_steps=*/3);  // b -> slot 1
  Scheduler sched(&engine, {kEos}, /*queue_limit=*/1);
  sched.submit(make_request("a", 5, 3));

  // WHEN the queue is full, try_submit sheds (no throw — this is load,
  // not an error) and submit throws QueueFullError; after a's admission
  // drains the queue, b lands and runs normally.
  require(!sched.try_submit(make_request("b", 5, 3)),
          "try_submit must shed when the queue is full");
  bool shed = false;
  try {
    sched.submit(make_request("b", 5, 3));
  } catch (const QueueFullError&) {
    shed = true;
  }
  require(shed, "submit must throw QueueFullError at the bound");
  require(sched.tick(), "tick admits a (draining the queue)");
  require(sched.try_submit(make_request("b", 5, 3)),
          "b lands once the queue has room");
  while (sched.tick()) {
  }
  require(sched.results().size() == 2,
          "exactly a and the landed b — a shed request is never recorded");
}

DGPP_TEST(scheduler_externalCancel_retiresAtNextTickWithoutExtraStep) {
  // GIVEN a mid-generation request (three tokens served, one to go),
  FakeEngine engine(/*slots=*/1, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.arm(0, {1, 2, 3, 4}, /*max_steps=*/4);
  Scheduler sched(&engine, {kEos});
  sched.submit(make_request("a", 5, 4));
  require(sched.tick(), "tick 1: admit + first step");
  require(sched.tick(), "tick 2: second step");

  // WHEN the client disconnects (external cancel) before the next tick,
  require(sched.cancel("a"), "cancel flags a live request");

  // THEN the next tick's SWEEP retires a — Cancelled with the tokens
  // served so far, NO fourth engine op (a cancelled request never pays
  // one more step), and the tick reports idle-after (the retirement
  // rode on the false — the documented tick contract). A late re-cancel
  // is a no-op.
  require(!sched.tick(), "the sweep tick retires a and reports idle-after");
  const std::string expected = "P:0:5 S:0:1 S:0:2 C:0";
  require(engine.op_stream() == expected,
          "cancel must retire without an extra step:\n  got:      " +
              engine.op_stream() + "\n  expected: " + expected);
  const auto& a = sched.results()[0];
  require(a.status == Scheduler::Result::Status::kCancelled &&
              a.reason == Scheduler::Result::Reason::kCancelled &&
              a.steps_done == 3 && ids_joined(a.generated) == "1,2,3",
          "a: Cancelled at 3 tokens");
  require(!sched.cancel("a"), "re-cancel of a terminal request is a no-op");
  require(!sched.has_pending(), "idle after the cancel");
  require(!sched.tick(), "a further tick is idle");
}

DGPP_TEST(scheduler_externalCancel_queuedRequestNeverAdmits) {
  // GIVEN a pool that fits one reservation at a time: a active, b queued,
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/2, /*block_tokens=*/4);
  engine.arm(0, {1, 2, 3}, /*max_steps=*/3);  // a -> slot 0
  engine.arm(1, {4, 5, 6}, /*max_steps=*/3);  // b -> slot 1 (never used)
  Scheduler sched(&engine, {kEos});
  sched.submit(make_request("a", 5, 3));
  sched.submit(make_request("b", 5, 3));
  require(sched.tick(), "tick 1 admits a; b defers on budget");

  // WHEN b is cancelled while queued,
  require(sched.cancel("b"), "cancel flags the queued request");
  while (sched.tick()) {
  }

  // THEN b retires Cancelled having never touched an engine slot (no
  // prefill, no close — the op stream is a's solo stream) and a
  // finishes untouched.
  const std::string expected = "P:0:5 S:0:1 S:0:2 C:0";
  require(engine.op_stream() == expected,
          "queued cancel must be invisible to the engine:\n  got:      " +
              engine.op_stream() + "\n  expected: " + expected);
  const auto& b = sched.results()[1];
  require(b.status == Scheduler::Result::Status::kCancelled &&
              b.steps_done == 0 && b.generated.empty() && b.slot == -1,
          "b: Cancelled, never admitted");
  require(sched.results()[0].status == Scheduler::Result::Status::kDone &&
              ids_joined(sched.results()[0].generated) == "1,2,3",
          "a finished normally");
}

DGPP_TEST(scheduler_observer_eventsInFireOrderWithLifecycle) {
  // GIVEN a scenario with EOS, a steps cap, and a cancellation (b reuses
  // slot 0 after a's EOS retires it — lowest-free-slot policy),
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.arm(0, {1, kEos}, /*max_steps=*/5);  // a: EOS at token 2
  engine.arm(0, {4, 5, 6}, /*max_steps=*/3);  // b: reuses slot 0
  Scheduler sched(&engine, {kEos});
  RecordingObserver observer;
  sched.set_observer(&observer);
  sched.submit(make_request("a", 5, 5));
  sched.submit(make_request("b", 5, 3, /*cancel_after=*/2));

  // WHEN run to completion,
  sched.run_to_completion();

  // THEN the observer saw every token BEFORE the retire that may follow
  // it, one retire per request, and the counts/steps match the results.
  const std::string expected =
      "T(a,1) T(a,999) E(a,2) T(b,4) T(b,5) E(b,2)";
  require(observer.replay() == expected,
          "event order drifted:\n  got:      " + observer.replay() +
              "\n  expected: " + expected);
  require(count_kind(observer, "end", "a") == 1 &&
              count_kind(observer, "end", "b") == 1,
          "exactly one retire event per request");
  const auto& a = sched.results()[0];
  const auto& b = sched.results()[1];
  require(a.reason == Scheduler::Result::Reason::kEos && a.steps_done == 2,
          "a: EOS at 2 tokens");
  require(b.status == Scheduler::Result::Status::kCancelled &&
              b.steps_done == 2,
          "b: cancelled after 2 tokens");
}

DGPP_TEST(scheduler_meters_trackQueueActiveTerminalAndTokens) {
  // GIVEN a bounded queue with one request landing,
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/2, /*block_tokens=*/4);
  engine.arm(0, {1, 2, 3}, /*max_steps=*/3);  // a
  engine.arm(0, {7, 8}, /*max_steps=*/2);     // b: admits after a retires
  Scheduler sched(&engine, {kEos});
  sched.submit(make_request("a", 5, 3));
  sched.submit(make_request("b", 5, 2));

  // WHEN sampled mid-run and after completion,
  const auto mid = sched.meters();  // everything queued pre-tick
  while (sched.tick()) {
  }
  const auto end = sched.meters();

  // THEN the meters agree with the policy at both points (pool meters
  // read the engine's own accounting, so in-use matches its arithmetic).
  require(mid.queued == 2 && mid.active == 0 && mid.terminal == 0 &&
              mid.tokens_generated == 0 && mid.pool_blocks_total == 2,
          "pre-tick meters: two queued, nothing moving");
  require(end.queued == 0 && end.active == 0 && end.terminal == 2 &&
              end.tokens_generated == 5 && end.pool_blocks_in_use == 0,
          "post-run meters: both terminal, 5 tokens total, pool drained");
}

DGPP_TEST(scheduler_retire_releasesEverythingButTheTombstoneAndResult) {
  // GIVEN three long-prompt requests on a batch scheduler (retired records
  // kept: the manifest contract), run to completion,
  FakeEngine engine(/*slots=*/1, /*total_blocks=*/1000, /*block_tokens=*/4);
  engine.arm(0, {1, 2, 3}, /*max_steps=*/3);  // a
  engine.arm(0, {4, 5, 6}, /*max_steps=*/3);  // b
  engine.arm(0, {7, 8, 9}, /*max_steps=*/3);  // c
  Scheduler sched(&engine, {kEos});
  sched.submit(make_request("a", 1000, 3));
  sched.submit(make_request("b", 1000, 3));
  sched.submit(make_request("c", 1000, 3));
  const auto queued = sched.meters();
  sched.run_to_completion();
  const auto done = sched.meters();

  // THEN the records held their prompts while queued (3,000 ids) and only
  // the results' generated ids after (9): a retired request releases its
  // prompt, keeps its result, and its tombstone still answers to its id.
  require(queued.records == 3 && queued.record_tokens == 3000,
          "queued records hold their prompts");
  require(done.records == 3 && done.terminal == 3 && done.record_tokens == 9,
          "retired records hold only the results' generated ids");
  require(sched.results().size() == 3 &&
              ids_joined(sched.results()[0].generated) == "1,2,3" &&
              ids_joined(sched.results()[2].generated) == "7,8,9",
          "the results are intact");
  require(!sched.cancel("a"), "a late cancel of a kept tombstone is a no-op");
  bool duplicate = false;
  try {
    sched.submit(make_request("a", 5, 3));
  } catch (const std::invalid_argument&) {
    duplicate = true;
  }
  require(duplicate, "a kept tombstone still rejects its id");
}

DGPP_TEST(scheduler_keepRetiredOff_compactsEveryTickAndMovesNoOp) {
  // GIVEN two identically armed two-row engines under two schedulers, one
  // keeping retired records (the batch contract), one not (the service's
  // and the peers' setting), and a script whose retirements interleave
  // with live requests: a and b overlap and b outlives a; c and d wait in
  // the queue, take slot 0 in turn, and d outlives b,
  const auto arm = [](FakeEngine& e) {
    e.arm(0, {1, 2, 3}, /*max_steps=*/3);             // a: slot 0
    e.arm(1, {10, 11, 12, 13, 14}, /*max_steps=*/5);  // b: slot 1
    e.arm(0, {20, 21}, /*max_steps=*/2);              // c: slot 0 after a
    e.arm(0, {30, 31, 32, 33}, /*max_steps=*/4);      // d: slot 0 after c
  };
  FakeEngine keep_engine(/*slots=*/2, /*total_blocks=*/100, /*block_tokens=*/4,
                         /*batch_capacity=*/2);
  FakeEngine drop_engine(/*slots=*/2, /*total_blocks=*/100, /*block_tokens=*/4,
                         /*batch_capacity=*/2);
  arm(keep_engine);
  arm(drop_engine);
  Scheduler keep(&keep_engine, {kEos});
  Scheduler drop(&drop_engine, {kEos});
  drop.set_keep_retired(false);
  for (Scheduler* s : {&keep, &drop}) {
    s->submit(make_request("a", 5, 3));
    s->submit(make_request("b", 5, 5));
    s->submit(make_request("c", 5, 2));
    s->submit(make_request("d", 5, 4));
  }

  // WHEN both tick to completion, the compacting one checked after every
  // tick,
  int ticks = 0;
  bool compacted_under_load = false;
  for (;;) {
    const bool more_keep = keep.tick();
    const bool more_drop = drop.tick();
    ++ticks;
    require(more_keep == more_drop, "the schedulers disagree on pending work");
    const auto k = keep.meters();
    const auto d = drop.meters();
    // THEN after every tick the compacting scheduler holds exactly its live
    // requests (results parallel to them, none terminal), while the other
    // keeps every record,
    require(d.records == d.active + d.queued,
            "tick " + std::to_string(ticks) + ": " + std::to_string(d.records) +
                " records held against " + std::to_string(d.active + d.queued) +
                " live");
    require(static_cast<int64_t>(drop.results().size()) == d.records &&
                std::none_of(drop.results().begin(), drop.results().end(),
                             [](const Scheduler::Result& r) {
                               return r.status == Scheduler::Result::Status::kDone ||
                                      r.status == Scheduler::Result::Status::kCancelled;
                             }),
            "tick " + std::to_string(ticks) + ": a retired result survived");
    require(k.records == 4 && k.terminal == d.terminal,
            "the batch scheduler keeps its four records; retirements agree");
    if (d.active > 0 && k.terminal > 0) compacted_under_load = true;
    if (!more_keep) break;
  }
  require(compacted_under_load, "the script must retire under live load");
  // the compaction moved no op — the same admissions, the same round-robin
  // slices and the same closes reached both engines — and the counters,
  // the transcripts and a late cancel read the same either way.
  require(drop_engine.op_stream() == keep_engine.op_stream(),
          "the op stream moved across a compaction:\n  kept:      " +
              keep_engine.op_stream() + "\n  compacted: " +
              drop_engine.op_stream());
  require(drop_engine.batch_calls() == keep_engine.batch_calls(),
          "the batch slices moved across a compaction");
  require(keep.results().size() == 4 &&
              ids_joined(keep.results()[1].generated) == "10,11,12,13,14" &&
              ids_joined(keep.results()[3].generated) == "30,31,32,33",
          "the batch scheduler's transcripts");
  require(drop.meters().records == 0 && drop.results().empty() &&
              drop.meters().terminal == 4 && drop.meters().tokens_generated == 14 &&
              keep.meters().tokens_generated == 14 && !drop.has_pending(),
          "drained: nothing held, four retirements and 14 tokens counted");
  require(!drop.cancel("a") && !keep.cancel("a"),
          "a late cancel is a no-op, compacted or kept");
}

// Grow-on-demand (M6 6d): the observer's growth events.
struct GrowLog final : dgpp::sched::SchedulerObserver {
  std::vector<std::string> grows, retires;
  void on_token(const std::string&, int64_t, int) override {}
  void on_retire(const std::string& id, const Scheduler::Result& r) override {
    retires.push_back(id + ":" + std::to_string(static_cast<int>(r.reason)) + ":" +
                      std::to_string(r.steps_done));
  }
  void on_grow(const std::string& id, int64_t tokens) override {
    grows.push_back(id + ":" + std::to_string(tokens));
  }
};

std::vector<int32_t> long_script(int n) {
  std::vector<int32_t> t;
  for (int i = 0; i < n; ++i) t.push_back(3 + (i % 5));  // never kEos
  return t;
}

DGPP_TEST(scheduler_growOnDemand_admitsEarlyGrowsAtTickTopAndShedsTheYoungest) {
  // Pool 6 blocks of 4 tokens; two requests of prompt 4 + 12 steps (16
  // tokens = 4 blocks each). Full-reserve admits one at a time. Grow with
  // a 4-token window reserves 8 tokens = 2 blocks each: both admitted at
  // once; each grows at tick top when its next step would write past its
  // reservation; when the pool cannot cover even the minimum the YOUNGEST
  // is shed (Done / kPoolExhausted) and the oldest completes its 12 steps.
  // Every decision is a function of scheduler state alone: two identical
  // runs produce identical engine op streams and observer logs.
  using Mode = dgpp::sched::AdmissionPolicy::Mode;
  const auto run = [](Mode mode, std::string* ops, GrowLog* log) {
    FakeEngine engine(2, 6, 4);
    engine.arm(0, long_script(12), 12);
    engine.arm(mode == Mode::kGrowOnDemand ? 1 : 0, long_script(12), 12);
    dgpp::sched::AdmissionPolicy policy;
    policy.mode = mode;
    policy.window_tokens = 4;
    Scheduler sched(&engine, {kEos}, 0, policy);
    sched.set_observer(log);
    sched.submit(make_request("a", 4, 12));
    sched.submit(make_request("b", 4, 12));
    sched.run_to_completion();
    *ops = engine.op_stream();
    return std::make_pair(sched.results(), sched.meters());
  };
  std::string ops1, ops2, ops_full;
  GrowLog log1, log2, log_full;
  const auto [res1, m1] = run(Mode::kGrowOnDemand, &ops1, &log1);
  const auto [res2, m2] = run(Mode::kGrowOnDemand, &ops2, &log2);
  const auto [resf, mf] = run(Mode::kFullReserve, &ops_full, &log_full);
  // Both admitted before either retired (P:1 precedes C:0).
  require(ops1.find("P:1") != std::string::npos &&
              ops1.find("P:1") < ops1.find("C:0"),
          "grow: b admitted while a is live: " + ops1);
  require(ops_full.find("C:0") < ops_full.find("P:0:") + 1 ||
              ops_full.find("C:0") < ops_full.rfind("P:0"),
          "full: b admitted only after a retired: " + ops_full);
  require(res1[0].status == Scheduler::Result::Status::kDone &&
              res1[0].reason == Scheduler::Result::Reason::kSteps &&
              res1[0].steps_done == 12,
          "grow: the oldest completes its 12 steps");
  require(res1[1].status == Scheduler::Result::Status::kDone &&
              res1[1].reason == Scheduler::Result::Reason::kPoolExhausted &&
              res1[1].steps_done < 12 && res1[1].steps_done > 0,
          "grow: the youngest shed at exhaustion after some tokens, got " +
              std::to_string(res1[1].steps_done));
  require(!log1.grows.empty() && m1.reservations_grown ==
                                     static_cast<int64_t>(log1.grows.size()),
          "grow: growth events observed and metered");
  require(m1.requests_shed_pool == 1 && m1.pool_blocks_in_use == 0,
          "grow: one shed, the pool empty at the end");
  require(resf[0].reason == Scheduler::Result::Reason::kSteps &&
              resf[1].reason == Scheduler::Result::Reason::kSteps &&
              log_full.grows.empty() && mf.requests_shed_pool == 0,
          "full-reserve: both complete, nothing grows or sheds");
  // Determinism: the same manifest, the same streams.
  require(ops1 == ops2 && log1.grows == log2.grows && log1.retires == log2.retires,
          "grow: two runs are identical");
  // The growth targets are monotone per request and never exceed the
  // lifetime (16 tokens); the first grow lands before the 5th step (the
  // 8-token reservation covers prompt 4 + 3 written tokens + 1).
  int64_t last_a = 0;
  for (const std::string& g : log1.grows) {
    if (g.rfind("a:", 0) != 0) continue;
    const int64_t t = std::stoll(g.substr(2));
    require(t > last_a && t <= 16, "grow: a's reservation grows monotonically within its lifetime");
    last_a = t;
  }
  require(last_a == 16, "grow: a reached its full reservation");
}

DGPP_TEST(scheduler_growOnDemand_aLoneRequestShedsItselfAndAWindowNeverBelowTheStepWidth) {
  // A single request that can never fit its lifetime is admitted under
  // grow (the door check is the service's) and sheds ITSELF when the pool
  // is spent: Done / kPoolExhausted with the tokens it got. And the
  // window is clamped to the engine's step width plus one, so a step
  // never writes past a reservation.
  FakeEngine engine(1, 3, 4);  // 12 tokens of pool
  engine.arm(0, long_script(40), 40);
  dgpp::sched::AdmissionPolicy policy;
  policy.mode = dgpp::sched::AdmissionPolicy::Mode::kGrowOnDemand;
  policy.window_tokens = 1;  // below the width + 1: clamped to 2
  Scheduler sched(&engine, {kEos}, 0, policy);
  sched.submit(make_request("solo", 4, 40));
  sched.run_to_completion();
  const Scheduler::Result& r = sched.results()[0];
  require(r.status == Scheduler::Result::Status::kDone &&
              r.reason == Scheduler::Result::Reason::kPoolExhausted,
          "solo: shed itself at exhaustion");
  // 12 tokens of pool: prompt 4 + the generated tokens the KV holds; the
  // pending pick is the last one — at most 9 tokens generated.
  require(r.steps_done >= 7 && r.steps_done <= 9,
          "solo: generated what the pool allowed, got " + std::to_string(r.steps_done));
  require(sched.meters().requests_shed_pool == 1 && sched.meters().pool_blocks_in_use == 0,
          "solo: metered and released");
  bool threw = false;
  try {
    dgpp::sched::AdmissionPolicy bad;
    bad.window_tokens = 0;
    Scheduler s(&engine, {kEos}, 0, bad);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "a zero window is refused");
}

}  // namespace

DGPP_TEST(scheduler_sampling_specArmsTheSlotBeforeItsPrefillPick) {
  // GIVEN a sampling-capable engine and one stochastic request,
  FakeEngine engine(/*slots=*/1, /*total_blocks=*/100, /*block_tokens=*/4,
                    /*batch_capacity=*/1, /*can_sample=*/true);
  engine.arm(0, {1, 2}, /*max_steps=*/2);
  Scheduler sched(&engine, {kEos});
  SchedulerRequest r = make_request("a", 5, 2);
  r.sampling.temperature = 1.0f;
  r.sampling.top_p = 0.95f;
  r.seed = 99;
  sched.submit(std::move(r));

  // WHEN it runs,
  sched.run_to_completion();

  // THEN the slot was armed with the request's seed immediately before its
  // prefill (the prefill pick is the first draw), and nothing else moved.
  require(engine.op_stream() == "A:0:99 P:0:5 S:0:1 C:0",
          "arming must precede the prefill: " + engine.op_stream());

  // AND a greedy request on the same engine arms nothing.
  FakeEngine greedy_engine(1, 100, 4, 1, /*can_sample=*/true);
  greedy_engine.arm(0, {1, 2}, 2);
  Scheduler greedy(&greedy_engine, {kEos});
  greedy.submit(make_request("g", 5, 2));
  greedy.run_to_completion();
  require(greedy_engine.op_stream() == "P:0:5 S:0:1 C:0",
          "greedy requests keep the exact op stream: " +
              greedy_engine.op_stream());

  // AND a constrained request arms its grammar after the sampling spec and
  // before the prefill — the prefill pick is the first masked position.
  FakeEngine constrained_engine(1, 100, 4, 1, /*can_sample=*/true);
  constrained_engine.arm(0, {1, 2}, 2);
  Scheduler constrained(&constrained_engine, {kEos});
  SchedulerRequest c = make_request("c", 5, 2);
  c.sampling.temperature = 1.0f;
  c.seed = 7;
  c.grammar.mode = dgpp::text::GrammarSpec::Mode::kRequired;
  c.grammar.tools.push_back(dgpp::text::GrammarTool{"f", false, {}, {}, {}, false});
  constrained.submit(std::move(c));
  constrained.run_to_completion();
  require(constrained_engine.op_stream() == "A:0:7 G:0:3:1 P:0:5 S:0:1 C:0",
          "the grammar arms before the prefill: " +
              constrained_engine.op_stream());
}

DGPP_TEST(scheduler_sampling_refusalsAreManifestErrors) {
  // A stochastic request against a greedy-only engine, and an invalid spec
  // against any engine, throw at submit — identically on every rank, never
  // reaching the engine.
  FakeEngine greedy_engine(1, 100, 4);
  Scheduler greedy(&greedy_engine, {kEos});
  SchedulerRequest stochastic = make_request("s", 5, 2);
  stochastic.sampling.temperature = 0.8f;
  bool threw = false;
  try {
    greedy.submit(stochastic);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "a greedy engine must refuse a stochastic request");
  require(greedy_engine.op_stream().empty(), "nothing reached the engine");

  // A constrained request against an engine without masks: refused at
  // submit too, and identically so on every rank.
  SchedulerRequest constrained = make_request("c", 5, 2);
  constrained.grammar.mode = dgpp::text::GrammarSpec::Mode::kForbidCalls;
  threw = false;
  try {
    greedy.submit(constrained);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "an engine without masks must refuse a constrained request");
  require(greedy_engine.op_stream().empty(), "nothing reached the engine");

  FakeEngine sampling_engine(1, 100, 4, 1, /*can_sample=*/true);
  Scheduler sampler(&sampling_engine, {kEos});
  SchedulerRequest bad = make_request("b", 5, 2);
  bad.sampling.temperature = 1.0f;
  bad.sampling.top_p = 2.0f;
  threw = false;
  try {
    sampler.submit(bad);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "an invalid spec must be refused at submit");
  require(!sampler.has_pending(), "the refused request was never queued");
}

// The logprobs channel: a request that asks arms the slot ("L") before the
// prefill, every token's entry reaches the observer right after its
// on_token — including the two tokens of a speculative step — and a
// request that asks on an engine that reports none is refused at submit.
class LogprobObserver : public dgpp::sched::SchedulerObserver {
 public:
  void on_token(const std::string& id, int64_t token, int steps_done) override {
    events.push_back("T:" + id + ":" + std::to_string(token) + ":" +
                     std::to_string(steps_done));
  }
  void on_token_logprobs(const std::string& id, int steps_done,
                         const dgpp::sample::Result& lp) override {
    events.push_back("L:" + id + ":" + std::to_string(lp.token) + ":" +
                     std::to_string(steps_done) + ":" +
                     std::to_string(lp.top_logprobs.size()));
  }
  void on_retire(const std::string&, const Scheduler::Result&) override {}
  std::vector<std::string> events;
};

DGPP_TEST(scheduler_logprobs_rideWithEveryTokenWhenAsked) {
  FakeEngine engine(1, 100, 4, 1, /*can_sample=*/true);
  engine.arm_batches(0, 10, {{11, 12}, {13}}, /*max_steps=*/4);
  Scheduler sched(&engine, {kEos});
  LogprobObserver observer;
  sched.set_observer(&observer);
  SchedulerRequest r = make_request("a", 5, 4);
  r.logprobs = 1;
  r.sampling.logprobs = 1;
  sched.submit(std::move(r));
  sched.run_to_completion();
  require(engine.op_stream() == "L:0 P:0:5 S:0:10 S:0:12 C:0",
          "arming precedes the prefill: " + engine.op_stream());
  const std::vector<std::string> want{
      "T:a:10:1", "L:a:10:1:1", "T:a:11:2", "L:a:11:2:1", "T:a:12:3",
      "L:a:12:3:1", "T:a:13:4", "L:a:13:4:1"};
  require(observer.events == want, "every token's logprobs follow its on_token");

  FakeEngine greedy_engine(1, 100, 4);
  Scheduler greedy(&greedy_engine, {kEos});
  SchedulerRequest ask = make_request("b", 5, 2);
  ask.logprobs = 0;
  ask.sampling.logprobs = 0;
  bool threw = false;
  try {
    greedy.submit(ask);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "an engine without logprobs refuses at submit");
}

// ---- the prefix cache (M7 stage B) ------------------------------------------
// The fake's geometry below: blocks of 4 tokens, pool alignment 4, chunks
// of 2048 (never reached), so a prompt's cuts are the aligned images of its
// boundaries only.

SchedulerRequest make_cached_request(const std::string& id,
                                     std::vector<int64_t> prompt,
                                     std::vector<int64_t> boundaries,
                                     int max_steps) {
  SchedulerRequest r;
  r.id = id;
  r.prompt = std::move(prompt);
  r.boundaries = std::move(boundaries);
  r.max_steps = max_steps;
  return r;
}

std::vector<int64_t> counted_prompt(int n, int64_t base = 100) {
  std::vector<int64_t> p(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) p[static_cast<size_t>(i)] = base + i;
  return p;
}

DGPP_TEST(scheduler_prefixCache_secondIdenticalPromptAttachesAtTheDeepestCut) {
  // GIVEN a 21-token prompt with boundaries at 5 and 13 (cuts 4 and 12) run
  // twice, back to back, on an engine with a four-slot arena:
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.set_prefix_arena(/*slots=*/4, /*align=*/4);
  engine.arm(0, {1, 2, 3}, 3);
  engine.arm(1, {1, 2, 3}, 3);
  Scheduler sched(&engine, {kEos});
  require(sched.prefix_slots() == 4, "the cache takes the engine's arena");
  sched.submit(make_cached_request("a", counted_prompt(21), {5, 13}, 3));
  sched.submit(make_cached_request("b", counted_prompt(21), {5, 13}, 3));
  sched.run_to_completion();

  // THEN the first prefill takes an entry at the deepest cut (12), the
  // second attaches to it there and takes none (the cut is the entry's);
  // both answers are the scripts'. The rolling snapshots: committed
  // positions 21, 22 — never aligned before the cap — so none.
  const std::string expected =
      "P:0:21 N:0:12@0 S:0:1 X:1:12@0 P:1:21 S:1:1 S:0:2 C:0 S:1:2 C:1";
  require(engine.op_stream() == expected,
          "prefix cache op stream:\n  got:      " + engine.op_stream() +
              "\n  expected: " + expected);
  const Scheduler::Meters m = sched.meters();
  require(m.prefix_hits == 1 && m.prefix_misses == 1 && m.prefix_snapshots == 1 &&
              m.prefix_tokens_saved == 12 && m.prefix_entries == 1 &&
              m.prefix_blocks_pinned == 3,
          "prefix meters: one miss, one hit of 12 tokens, one entry pinning 3 blocks");
  require(ids_joined(sched.results()[1].generated) == "1,2,3", "b's ids");
}

DGPP_TEST(scheduler_prefixCache_reusesLongDocumentWithChangedTail) {
  FakeEngine engine(1, 10000, 64);
  engine.set_prefix_arena(6, 4);
  Scheduler sched(&engine, {});
  const auto original = counted_prompt(8221);
  auto changed = original;
  for (size_t i = 8192; i < changed.size(); ++i) changed[i] += 100000;
  for (int i = 0; i < 3; ++i) {
    engine.arm(0, {10}, 1);
    sched.submit(make_cached_request(std::to_string(i), i == 0 ? original : changed, {8220}, 1));
    sched.run_to_completion();
    if (i == 1)
      require(sched.meters().prefix_tokens_saved == 6144,
              "a changed question must reuse the earlier shared-document snapshot");
  }
  require(sched.meters().prefix_hits == 2 && sched.meters().prefix_misses == 1,
          "changed tail and identical repeat both hit");
  require(sched.meters().prefix_tokens_saved == 6144 + 8220,
          "identical repeats still use the deepest cut");
}

DGPP_TEST(scheduler_prefixCache_documentPolicyUnderArenaPressure) {
  for (int slots : {1, 2, 6, 27}) {
    FakeEngine engine(1, 10000, 4);
    engine.set_prefix_arena(slots, 4, 8);
    Scheduler sched(&engine, {});
    for (int i = 0; i < 7; ++i) {
      auto prompt = counted_prompt(39);
      prompt[35] += 100 * i;
      engine.arm(0, {10}, 1);
      sched.submit(make_cached_request(std::to_string(i), prompt, {36}, 1));
      sched.run_to_completion();
    }
    const auto meters = sched.meters();
    require(meters.prefix_entries <= slots, "document policy obeys arena capacity");
    if (slots >= 2)
      require(meters.prefix_hits == 6 && meters.prefix_tokens_saved == 6 * 24,
              "one shared body survives seven question variants under LRU pressure");
    else
      require(meters.prefix_hits == 0 && meters.prefix_snapshots == 7,
              "one-slot arena keeps the original deepest-snapshot behavior");
    auto edited = counted_prompt(39);
    edited[23] += 1000;
    engine.arm(0, {10}, 1);
    sched.submit(make_cached_request("body-edit", edited, {36}, 1));
    sched.run_to_completion();
    require(sched.meters().prefix_misses == meters.prefix_misses + 1,
            "editing before the body cut must miss");
  }
}

DGPP_TEST(scheduler_prefixCache_documentPolicyLeavesShortAndOptedOutPromptsAlone) {
  FakeEngine engine(1, 1000, 4);
  engine.set_prefix_arena(6, 4, 8);
  Scheduler sched(&engine, {});
  engine.arm(0, {10}, 1);
  sched.submit(make_cached_request("short", counted_prompt(31), {28}, 1));
  sched.run_to_completion();
  require(sched.meters().prefix_snapshots == 1, "fewer than four chunks: only the deepest cut");
  auto uncached = make_cached_request("off", counted_prompt(39), {36}, 1);
  uncached.no_cache = true;
  engine.arm(0, {10}, 1);
  sched.submit(uncached);
  sched.run_to_completion();
  require(sched.meters().prefix_snapshots == 1 && sched.meters().prefix_misses == 1,
          "opted-out long prompt never uses the arena");
}

DGPP_TEST(scheduler_prefixCache_documentContinuationPublishesBothSnapshots) {
  ChunkFakeEngine engine(1000);
  engine.set_prefix_arena(6, 2, 8);
  dgpp::sched::AdmissionPolicy policy;
  policy.prefill_budget_tokens = 8;
  Scheduler sched(&engine, {}, 0, policy);
  auto prompt = counted_prompt(39);
  for (int i = 0; i < 3; ++i) {
    engine.arm(0, {10}, 1);
    if (i == 1) prompt[36] += 1000;
    sched.submit(make_cached_request(std::to_string(i), prompt, {38}, 1));
    sched.run_to_completion();
  }
  require(sched.meters().prefix_hits == 2 && sched.meters().prefix_tokens_saved == 24 + 38,
          "continuations publish the body and final cuts for later requests");
}

DGPP_TEST(scheduler_prefixCache_mtpLookaheadRejectsAnEditExactlyAtTheCut) {
  FakeEngine engine(1, 1000, 4);
  engine.set_prefix_arena(8, 4, 8);
  engine.prefill_lookahead_ = true;
  Scheduler sched(&engine, {});
  for (int i = 0; i < 3; ++i) {
    auto prompt = counted_prompt(39);
    if (i == 1) prompt[24] += 1000;
    if (i == 2) prompt[25] += 2000;
    engine.arm(0, {10}, 1);
    sched.submit(make_cached_request(std::to_string(i), prompt, {36}, 1));
    sched.run_to_completion();
    if (i == 1)
      require(sched.meters().prefix_hits == 0, "changed lookahead must not attach stale MTP state");
  }
  require(sched.meters().prefix_hits == 1 && sched.meters().prefix_tokens_saved == 24,
          "entries with distinct lookahead tokens coexist; a later tail edit reuses the right one");
}

DGPP_TEST(prefixCache_lookaheadAndRollingEntriesDedupeSafely) {
  using dgpp::sched::PrefixCache;
  PrefixCache cache({4, 4, 8});
  auto ids = counted_prompt(17);
  const auto h = PrefixCache::hash_prefix(ids.data(), 8);
  require(cache.insert(ids.data(), 8, cache.take_free_slot(), 1, {}, ids[8]) >= 0, "prefill entry");
  require(cache.lookup(ids, {8}, {h}) >= 0, "unchanged lookahead hits");
  ids[8] += 1;
  require(cache.lookup(ids, {8}, {h}) < 0, "changed lookahead misses");
  const int rolling = cache.insert(ids.data(), 8, cache.take_free_slot(), 2);
  require(rolling >= 0,
          "unconstrained rolling entry is not discarded for constrained prefill state");
  require(cache.lookup(ids, {8}, {h}) == rolling,
          "rolling state can catch up to a different next token");
  const int duplicate_slot = cache.take_free_slot();
  require(cache.insert(ids.data(), 8, duplicate_slot, 3, {}, ids[8]) < 0,
          "unconstrained state safely supersedes a new constrained duplicate");
  cache.give_back_slot(duplicate_slot);
}

DGPP_TEST(scheduler_prefixCache_documentContinuationCancelAndFailureReleaseBothSlots) {
  class FailingEngine : public ChunkFakeEngine {
   public:
    FailingEngine() : ChunkFakeEngine(1000) {}
    bool fail = false;
    PrefillProgress advance_prefill(int req, int64_t budget = 0) override {
      if (fail) {
        fail = false;
        throw std::runtime_error("injected continuation failure");
      }
      return ChunkFakeEngine::advance_prefill(req, budget);
    }
  };
  for (bool fail : {false, true}) {
    FailingEngine engine;
    engine.set_prefix_arena(2, 2, 8);
    engine.arm(0, {10}, 1);
    engine.arm(0, {20}, 1);
    dgpp::sched::AdmissionPolicy policy;
    policy.prefill_budget_tokens = 8;
    Scheduler sched(&engine, {}, 0, policy);
    sched.submit(make_cached_request("unfinished", counted_prompt(39), {38}, 1));
    sched.tick();
    require(sched.meters().prefix_entries == 0, "unfinished prefill cannot publish snapshots");
    if (fail) {
      engine.fail = true;
      bool threw = false;
      try {
        sched.tick();
      } catch (const std::runtime_error&) {
        threw = true;
      }
      require(threw, "engine failures propagate to the serving loop");
    }
    require(sched.cancel("unfinished"), "cancel pending/failed prefill");
    sched.run_to_completion();
    require(engine.pinned_blocks() == 0 && sched.meters().pool_blocks_in_use == 0,
            "failed/cancelled prefill releases both snapshots and its live reservation");
    sched.submit(make_cached_request("reuse", counted_prompt(39), {38}, 1));
    sched.run_to_completion();
    require(sched.meters().prefix_snapshots == 2, "both arena slots are available after cleanup");
  }
}

DGPP_TEST(scheduler_prefixCache_syncFailureAfterBodySnapshotReleasesItsBlocks) {
  class FailingEngine : public FakeEngine {
   public:
    FailingEngine() : FakeEngine(1, 1000, 4) {}
    bool fail = true;
    int32_t prefill_cached(int req, const std::vector<int64_t>& prompt,
                           PrefixPrefill* plan) override {
      if (!fail) return FakeEngine::prefill_cached(req, prompt, plan);
      fail = false;
      auto partial = *plan;
      partial.snap_slot = -1;
      FakeEngine::prefill_cached(req, prompt, &partial);
      live_.erase(req);  // the real adapter unwinds an unreserved failed prefill
      throw std::runtime_error("injected failure after earlier snapshot");
    }
  } engine;
  engine.set_prefix_arena(2, 4, 8);
  engine.arm(0, {10}, 1);
  engine.arm(0, {20}, 1);
  Scheduler sched(&engine, {});
  sched.submit(make_cached_request("failed", counted_prompt(39), {36}, 1));
  bool threw = false;
  try {
    sched.tick();
  } catch (const std::runtime_error&) {
    threw = true;
  }
  require(threw, "prefill failure propagates");
  require(sched.cancel("failed"), "discard the failed admission");
  sched.run_to_completion();
  require(engine.pinned_blocks() == 0 && sched.meters().pool_blocks_in_use == 0,
          "early snapshot is released even when final snapshot never ran");
  sched.submit(make_cached_request("reuse", counted_prompt(39), {36}, 1));
  sched.run_to_completion();
  require(sched.meters().prefix_snapshots == 2, "both slots returned after sync failure");
}

DGPP_TEST(prefixCache_nearestNamesWhereAMissedPromptDiverges) {
  // The miss diagnostic: among the live entries, the one
  // sharing the longest prefix with a prompt, and that length.
  using dgpp::sched::PrefixCache;
  PrefixCache::Config cfg;
  cfg.slots = 4;
  cfg.align = 4;
  PrefixCache cache(cfg);
  require(cache.nearest(counted_prompt(8)).entry < 0, "an empty cache names nothing");
  const std::vector<int64_t> a = counted_prompt(40);
  const int e0 = cache.insert(a.data(), 12, cache.take_free_slot(), 1);
  const int e1 = cache.insert(a.data(), 36, cache.take_free_slot(), 2);
  require(e0 >= 0 && e1 >= 0, "two entries over the same sequence");
  std::vector<int64_t> edited = a;
  for (size_t i = 20; i < edited.size(); ++i) edited[i] += 1000;
  PrefixCache::Nearest n = cache.nearest(edited);
  require(n.entry == e1 && n.common == 20, "the deeper entry shares 20 tokens: the edit is at 20");
  std::vector<int64_t> first = a;
  first[0] += 1000;
  n = cache.nearest(first);
  require(n.common == 0 && n.entry == e1, "nothing shared: the deeper entry named, common 0");
  std::vector<int64_t> longer = a;
  longer.push_back(900);
  n = cache.nearest(longer);
  require(n.entry == e1 && n.common == 36, "a prompt extending the entry shares all of it");
  std::vector<int64_t> shorter(a.begin(), a.begin() + 10);
  n = cache.nearest(shorter);
  require(n.common == 10 && n.entry == e1, "a shorter prompt shares its own length; the deeper among equals");
  // The ghosts: an evicted entry is still named at the prompt's cut.
  const std::vector<int64_t> cuts = {12, 36};
  const std::vector<uint64_t> hashes = {PrefixCache::hash_prefix(a.data(), 12),
                                        PrefixCache::hash_prefix(a.data(), 36)};
  require(cache.ghost_at(cuts, hashes).position == 0, "nothing evicted yet");
  require(cache.evict_lru() >= 0, "the LRU entry (e0, used at 1) goes");
  PrefixCache::Ghost g = cache.ghost_at(cuts, hashes);
  require(g.position == 12 && g.eviction == 1 && g.last_use == 1, "the ghost at cut 12");
  require(cache.evict_lru() >= 0, "then e1");
  g = cache.ghost_at(cuts, hashes);
  require(g.position == 36 && g.eviction == 2, "the deepest cut's ghost is named first");
  require(cache.ghost_at({12}, {PrefixCache::hash_prefix(edited.data(), 12)}).position == 12 &&
              cache.ghost_at({20}, {PrefixCache::hash_prefix(edited.data(), 20)}).position == 0,
          "a ghost answers by prefix hash and position, not by the prompt's tail");
}

DGPP_TEST(scheduler_prefixCache_twoInterleavedConversationsKeepHittingUnderATightArena) {
  // Two multi-turn conversations advancing in lockstep on a two-slot engine
  // with a six-slot arena (2026-09-07: the agent session's pattern —
  // several streams, each turn extending its predecessor's prompt). A
  // echoes its answers verbatim into the next prompt (a client returning
  // the reasoning: the turn attaches at the CLOSE entry and prefills only
  // the new message); B rewrites the answer's tokens (a client stripping
  // the reasoning: the turn attaches at the previous prompt's deepest cut,
  // its assistant header, and re-prefills the rewritten answer too). Every
  // turn after the first must hit even as the arena turns over: four
  // entries per round against six slots, two of them the live rolling
  // snapshots.
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/400, /*block_tokens=*/4);
  engine.set_prefix_arena(/*slots=*/6, /*align=*/4);
  Scheduler sched(&engine, {kEos});
  constexpr int kHeader = 7;  // the assistant header: a boundary at the prompt's end
  struct Conv {
    std::string name;
    bool echo;
    std::vector<int64_t> prompt;
    std::vector<int64_t> boundaries;
    int64_t base;
  };
  const auto opening = [&](Conv* c, int64_t base) {
    c->base = base;
    c->prompt = counted_prompt(21, base);     // system + user
    c->boundaries = {5, 13};
    c->prompt.push_back(kHeader);             // <|assistant|>
    c->boundaries.push_back(static_cast<int64_t>(c->prompt.size()) - 1);
  };
  Conv A{"A", true, {}, {}, 0}, B{"B", false, {}, {}, 0};
  opening(&A, 100);
  opening(&B, 5000);
  const auto align = [](int64_t x) { return (x / 4) * 4; };
  std::vector<int64_t> expect_attach;  // per turn, the positions both must attach at
  int64_t gen_serial = 1000;
  for (int turn = 0; turn < 4; ++turn) {
    // The scripts: five answer tokens then EOS, distinct per turn.
    std::vector<int> script;
    for (int i = 0; i < 5; ++i) script.push_back(static_cast<int>(gen_serial) + i);
    gen_serial += 10;
    script.push_back(static_cast<int>(kEos));
    engine.arm(0, script, 8);
    engine.arm(1, script, 8);
    const std::string ia = A.name + std::to_string(turn), ib = B.name + std::to_string(turn);
    sched.submit(make_cached_request(ia, A.prompt, A.boundaries, 8));
    sched.submit(make_cached_request(ib, B.prompt, B.boundaries, 8));
    sched.run_to_completion();
    // The next prompts: the answer (echoed, or rewritten for B), the EOS
    // as the next message's marker, four new tokens, the header.
    for (Conv* c : {&A, &B}) {
      const int64_t prev_len = static_cast<int64_t>(c->prompt.size());
      const int64_t eos_at = prev_len + 5;
      // The expectation for the turn to come: A at the close entry
      // (aligned image of the EOS position), B at the previous prompt's
      // header cut.
      expect_attach.push_back(c->echo ? align(eos_at) : align(prev_len - 1));
      for (int i = 0; i < 5; ++i)
        c->prompt.push_back(c->echo ? script[static_cast<size_t>(i)]
                                    : script[static_cast<size_t>(i)] + 777);
      c->prompt.push_back(kEos);
      c->boundaries.push_back(eos_at);
      for (int i = 0; i < 4; ++i) c->prompt.push_back(c->base + 300 + turn * 4 + i);
      c->prompt.push_back(kHeader);
      c->boundaries.push_back(static_cast<int64_t>(c->prompt.size()) - 1);
    }
  }
  // Every turn after the first attached, at the expected position.
  const Scheduler::Meters m = sched.meters();
  require(m.prefix_misses == 2 && m.prefix_hits == 6,
          "two cold openings, six hits: misses " + std::to_string(m.prefix_misses) +
              " hits " + std::to_string(m.prefix_hits));
  std::vector<int64_t> attached;
  {
    std::istringstream ops(engine.op_stream());
    std::string op;
    while (ops >> op)
      if (op.rfind("X:", 0) == 0) {
        const size_t colon = op.find(':', 2);
        attached.push_back(std::stoll(op.substr(colon + 1, op.find('@') - colon - 1)));
      }
  }
  std::vector<int64_t> want(expect_attach.begin(), expect_attach.end() - 2);  // the last round's expectation is for a turn never sent
  std::sort(attached.begin(), attached.end());
  std::sort(want.begin(), want.end());
  std::string got_s, want_s;
  for (const int64_t x : attached) got_s += std::to_string(x) + " ";
  for (const int64_t x : want) want_s += std::to_string(x) + " ";
  require(attached == want, "attach positions: got " + got_s + "; expected " + want_s);
  require(m.prefix_evictions > 0, "the arena turned over: " + std::to_string(m.prefix_evictions));
}

DGPP_TEST(scheduler_prefixCache_rollingSnapshotBecomesTheCloseEntryTheNextTurnAttachesTo) {
  // GIVEN a 21-token prompt whose answer runs six tokens and ends in EOS:
  // committed positions 21 (after the prefill pick), 22, 23, 24 <- aligned,
  // 25, 26 (the EOS pending) — one rolling snapshot at 24; at retire the
  // entry sits at floor((27 - 1) / 4) * 4 = 24, the aligned image of the
  // EOS token's position 26.
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.set_prefix_arena(/*slots=*/4, /*align=*/4);
  engine.arm(0, {31, 32, 33, 34, 35, kEos}, 8);
  Scheduler sched(&engine, {kEos});
  const std::vector<int64_t> prompt = counted_prompt(21);
  sched.submit(make_cached_request("a", prompt, {5, 13}, 8));
  sched.run_to_completion();
  const std::string expected_a =
      "P:0:21 N:0:12@0 S:0:31 S:0:32 S:0:33 RS:0:24@1 S:0:34 S:0:35 C:0";
  require(engine.op_stream() == expected_a,
          "turn one:\n  got:      " + engine.op_stream() + "\n  expected: " + expected_a);
  Scheduler::Meters m = sched.meters();
  require(m.prefix_rolling == 1 && m.prefix_close_entries == 1 && m.prefix_entries == 2,
          "one rolling snapshot became the close entry; two entries live");

  // WHEN the next turn arrives: the conversation so far (the prompt, the
  // five answer tokens, the EOS as the next message's marker) and a new
  // message, with boundaries at the markers (5, 13, 26 -> cuts 4, 12, 24):
  std::vector<int64_t> next = prompt;
  next.insert(next.end(), {31, 32, 33, 34, 35, kEos});
  next.insert(next.end(), {200, 201, 202, 203});
  engine.arm(0, {41, 42}, 2);
  sched.submit(make_cached_request("b", next, {5, 13, 26}, 2));
  sched.run_to_completion();

  // THEN it attaches at 24 (the close entry, the deepest cut) and prefills
  // the seven remaining tokens; no new entry at a cut (24 is the deepest),
  // and its own close entry at 32 — the retire-time snapshot, its committed
  // position 31 + 2 - 1 being aligned.
  const std::string expected_b = expected_a + " X:0:24@1 P:0:31 S:0:41 RS:0:32@2 C:0";
  require(engine.op_stream() == expected_b,
          "turn two:\n  got:      " + engine.op_stream() + "\n  expected: " + expected_b);
  m = sched.meters();
  require(m.prefix_hits == 1 && m.prefix_tokens_saved == 24, "the second turn hit at 24");
}

DGPP_TEST(scheduler_prefixCache_retireAtAnAlignedPositionSnapshotsTheLiveState) {
  // GIVEN a 21-token prompt whose answer ends at the cap after four tokens:
  // committed positions 21, 22, 23 — never aligned before the cap — and 24
  // at retire, aligned. THEN the retire takes the snapshot from the live
  // slot (before the close) and it becomes the close entry at 24, where
  // the next turn (a marker at the fourth answer token's position, 24)
  // cuts and attaches.
  FakeEngine engine(/*slots=*/1, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.set_prefix_arena(/*slots=*/4, /*align=*/4);
  engine.arm(0, {31, 32, 33, 34}, 4);
  engine.arm(0, {41}, 1);
  Scheduler sched(&engine, {kEos});
  const std::vector<int64_t> prompt = counted_prompt(21);
  sched.submit(make_cached_request("a", prompt, {5, 13}, 4));
  sched.run_to_completion();
  const std::string expected_a = "P:0:21 N:0:12@0 S:0:31 S:0:32 S:0:33 RS:0:24@1 C:0";
  require(engine.op_stream() == expected_a,
          "retire-time snapshot:\n  got:      " + engine.op_stream() +
              "\n  expected: " + expected_a);
  require(sched.meters().prefix_close_entries == 1, "the close entry at 24");
  std::vector<int64_t> next = prompt;
  next.insert(next.end(), {31, 32, 33, kEos, 200, 201});
  sched.submit(make_cached_request("b", next, {5, 13, 24}, 1));
  sched.run_to_completion();
  const std::string expected_b = expected_a + " X:0:24@1 P:0:27 C:0";
  require(engine.op_stream() == expected_b,
          "the next turn attaches at 24:\n  got:      " + engine.op_stream() +
              "\n  expected: " + expected_b);
}

DGPP_TEST(scheduler_prefixCache_aStepCutShortByTheCapTakesNoRetireSnapshot) {
  // A two-token step whose first token completes the answer (the cap at
  // four tokens: 31, [32,33], [34,35] — 35 dropped) leaves the engine's
  // state one row past the committed position; the retire-time snapshot
  // cannot be taken from the live slot there (the fake, like the real
  // arena, refuses a position that is not its own). With an align of 1
  // (GLM-4.7: every position is snapshot-eligible) the rolling entry at 23
  // stands as the close entry and no snapshot is attempted at 24.
  {
    FakeEngine engine(/*slots=*/1, /*total_blocks=*/100, /*block_tokens=*/4);
    engine.set_prefix_arena(/*slots=*/4, /*align=*/1);
    engine.set_step_tokens_max(2);
    engine.arm_batches(0, 31, {{32, 33}, {34, 35}}, 4);
    Scheduler sched(&engine, {kEos});
    sched.submit(make_cached_request("a", counted_prompt(21), {5, 13}, 4));
    sched.run_to_completion();
    const std::string ops = engine.op_stream();
    require(ops.find("RS:0:23@") != std::string::npos, "align 1: the rolling snapshot at 23 was taken: " + ops);
    require(ops.find("RS:0:24@") == std::string::npos, "align 1: no retire-time snapshot at 24: " + ops);
    require(ops.size() >= 3 && ops.compare(ops.size() - 3, 3, "C:0") == 0, "align 1: the slot closed: " + ops);
    require(sched.meters().prefix_close_entries == 1, "align 1: the rolling entry at 23 is the close entry");
  }
  // With an align of 4 the cut lands exactly on the aligned 24 (21 + 4 - 1):
  // the old retire path asked for the live snapshot there and the arena
  // refused (the fabric's first GLM-4.7 request, 2026-09-10).
  {
    FakeEngine engine(/*slots=*/1, /*total_blocks=*/100, /*block_tokens=*/4);
    engine.set_prefix_arena(/*slots=*/4, /*align=*/4);
    engine.set_step_tokens_max(2);
    engine.arm_batches(0, 31, {{32, 33}, {34, 35}}, 4);
    Scheduler sched(&engine, {kEos});
    sched.submit(make_cached_request("a", counted_prompt(21), {5, 13}, 4));
    sched.run_to_completion();
    const std::string ops = engine.op_stream();
    require(ops.find("RS:0:24@") == std::string::npos, "align 4: no retire-time snapshot at 24: " + ops);
    require(ops.size() >= 3 && ops.compare(ops.size() - 3, 3, "C:0") == 0, "align 4: the slot closed: " + ops);
  }
}

DGPP_TEST(scheduler_prefixCache_twoTokenStepsHopOverAnAlignedPositionAndSnapshotIt) {
  // The MTP graph's two-token steps (M7's measured limit, closed
  // 2026-09-05): a 21-token prompt's committed count runs 21, 23, 25 —
  // never ON 24 — so no rolling snapshot could land there. The scheduler
  // arms the engine at 23 for the hop over 24 (the fake takes the state
  // after the step's first row when the step commits two), and the hop
  // becomes the close entry the next turn (a marker at 24) attaches to.
  FakeEngine engine(/*slots=*/1, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.set_prefix_arena(/*slots=*/4, /*align=*/4);
  engine.set_step_tokens_max(2);
  engine.arm_batches(0, 31, {{32, 33}, {34, 35}, {36, 37}}, 6);
  engine.arm(0, {41}, 1);
  Scheduler sched(&engine, {kEos});
  const std::vector<int64_t> prompt = counted_prompt(21);
  sched.submit(make_cached_request("a", prompt, {5, 13}, 6));
  sched.run_to_completion();
  // Committed 21 (22 is not aligned: no arm) → [32,33] → 23, armed for 24 →
  // [34,35] commits two: the hop at 24 → 25 (26 not aligned) → [36,37] →
  // the cap at six tokens (37 dropped) → retire with committed 26: the
  // close entry is the hop's, at 24.
  const std::string expected_a = "P:0:21 N:0:12@0 S:0:31 H:0:24@1 S:0:33 S:0:35 C:0";
  require(engine.op_stream() == expected_a,
          "the hop:\n  got:      " + engine.op_stream() +
              "\n  expected: " + expected_a);
  require(sched.meters().prefix_hops == 1 && sched.meters().prefix_rolling == 1 &&
              sched.meters().prefix_close_entries == 1,
          "one hop, counted as a rolling snapshot, one close entry");
  std::vector<int64_t> next = prompt;
  next.insert(next.end(), {31, 32, 33, kEos, 200, 201});
  sched.submit(make_cached_request("b", next, {5, 13, 24}, 1));
  sched.run_to_completion();
  const std::string expected_b = expected_a + " X:0:24@1 P:0:27 C:0";
  require(engine.op_stream() == expected_b,
          "the next turn attaches at the hop's entry:\n  got:      " + engine.op_stream() +
              "\n  expected: " + expected_b);
  // A one-token engine never arms (the plain stream), and a two-token step
  // that commits one token lands on the position: the regular rolling
  // snapshot takes it at the next tick.
  FakeEngine one(/*slots=*/1, /*total_blocks=*/100, /*block_tokens=*/4);
  one.set_prefix_arena(/*slots=*/4, /*align=*/4);
  one.set_step_tokens_max(2);
  one.arm_batches(0, 31, {{32, 33}, {34}, {35, 36}}, 6);
  Scheduler sched1(&one, {kEos});
  sched1.submit(make_cached_request("a", prompt, {5, 13}, 6));
  sched1.run_to_completion();
  // 21 → [32,33] → 23 (armed for 24) → [34] commits one: 24, no hop → the
  // rolling snapshot at 24 before the next step → [35,36] → the cap.
  const std::string expected_one = "P:0:21 N:0:12@0 S:0:31 S:0:33 RS:0:24@1 S:0:34 C:0";
  require(one.op_stream() == expected_one,
          "a one-token step lands on the position:\n  got:      " + one.op_stream() +
              "\n  expected: " + expected_one);
  require(sched1.meters().prefix_hops == 0 && sched1.meters().prefix_rolling == 1,
          "no hop, one rolling snapshot");
}

DGPP_TEST(scheduler_prefixCache_admissionCountsTheSnapshotCopiesAgainstThePool) {
  // The prefix-cache curve sweep of 2026-09-05 (43 slots, eight
  // conversations, a 4,096-token pool) killed the service:
  // session_reserve_blocks found the pool one block short after the prefill.
  // An entry's snapshot takes a private copy of its partial block, and so
  // does a request's rolling snapshot; the admission counted neither. It now
  // counts the cut entry's copy and one block of headroom for the rolling
  // copy (a position aligned to kpool but not to the block), and a rolling
  // snapshot that finds no block evicts an unattached entry or skips.
  // GIVEN a 5-block pool of 8-token blocks at kpool 4, and a 17-token
  // prompt with a boundary at 13 (the cut at 12: a partial block) and six
  // steps (23 tokens: 3 blocks reserved): the admission needs 3 + 1 (the
  // cut's copy) + 1 (headroom) = 5 blocks — exactly the pool.
  FakeEngine engine(/*slots=*/1, /*total_blocks=*/5, /*block_tokens=*/8);
  engine.set_partial_pins(true);
  engine.set_prefix_arena(/*slots=*/4, /*align=*/4);
  engine.arm(0, {31, 32, 33, 34, 35, 36}, 6);
  Scheduler sched(&engine, {kEos});
  const std::vector<int64_t> prompt = counted_prompt(17);
  sched.submit(make_cached_request("a", prompt, {5, 13}, 6));
  sched.run_to_completion();
  // Committed 17..22: the rolling snapshot at 20 (a partial block) took the
  // headroom block; nothing overdrew the pool (the fake would have thrown).
  const std::string expected_a =
      "P:0:17 N:0:12@0 S:0:31 S:0:32 S:0:33 RS:0:20@1 S:0:34 S:0:35 C:0";
  require(engine.op_stream() == expected_a,
          "the admission left room for both copies:\n  got:      " + engine.op_stream() +
              "\n  expected: " + expected_a);
  require(sched.meters().prefix_close_entries == 1 &&
              sched.meters().prefix_skipped_no_block == 0,
          "the close entry at 20, no snapshot skipped for blocks");
  // AND a different prompt while a's two entries pin two blocks: the
  // admission needs 5 free of 3 — both entries give way (the cut entry
  // first, the older), then the same sequence in the same slots.
  engine.arm(0, {31, 32, 33, 34, 35, 36}, 6);
  sched.submit(make_cached_request("c", counted_prompt(17, 300), {5, 13}, 6));
  sched.run_to_completion();
  const std::string expected_c =
      expected_a + " F:0 F:1 P:0:17 N:0:12@0 S:0:31 S:0:32 S:0:33 RS:0:20@1 S:0:34 S:0:35 C:0";
  require(engine.op_stream() == expected_c,
          "entries give their blocks to an admission that needs them:\n  got:      " +
              engine.op_stream() + "\n  expected: " + expected_c);
  require(sched.meters().prefix_evictions == 2, "two evictions for the blocks");
  // AND a 4-block pool: the one-step request (3 blocks + the cache's 2 = 5)
  // cannot carry the cache; it admits without it.
  FakeEngine tight(/*slots=*/2, /*total_blocks=*/4, /*block_tokens=*/8);
  tight.set_partial_pins(true);
  tight.set_prefix_arena(/*slots=*/4, /*align=*/4);
  tight.arm(0, {41}, 1);
  Scheduler sched2(&tight, {kEos});
  sched2.submit(make_cached_request("d", prompt, {5, 13}, 1));
  sched2.run_to_completion();
  // (Three blocks of reservation plus the cache's two exceed the pool, so
  // the request runs without the cache — never a deadlock, never an
  // overdraw.)
  require(tight.op_stream() == "P:0:17 C:0",
          "a pool too small for the cache's blocks admits the request cache-less: " +
              tight.op_stream());
}

DGPP_TEST(scheduler_prefixCache_theAttachTargetSurvivesTheSnapshotSlotsEviction) {
  // The curve sweep's find (2026-09-05, "PrefixArena: slot 25 is empty"):
  // with the arena full, an admission that attaches to one entry and takes
  // a new entry at a deeper cut acquired the snapshot slot by evicting the
  // LRU unattached entry — its own attach target when that was the oldest —
  // and then prefilled from an empty slot. The attach now happens before
  // the acquisition, so the target is protected and the other entry gives
  // way. GIVEN a two-slot arena filled by a's cut entry (12, the older) and
  // its close entry (24), and b sharing a's prompt but not its answer, so it
  // attaches at 12 and snapshots at 24:
  FakeEngine engine(/*slots=*/1, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.set_prefix_arena(/*slots=*/2, /*align=*/4);
  engine.arm(0, {31, 32, 33, 34}, 4);
  engine.arm(0, {41}, 1);
  Scheduler sched(&engine, {kEos});
  const std::vector<int64_t> prompt = counted_prompt(21);
  sched.submit(make_cached_request("a", prompt, {5, 13}, 4));
  sched.run_to_completion();
  const std::string expected_a = "P:0:21 N:0:12@0 S:0:31 S:0:32 S:0:33 RS:0:24@1 C:0";
  require(engine.op_stream() == expected_a, "a fills the arena: " + engine.op_stream());
  std::vector<int64_t> other = prompt;
  other.insert(other.end(), {200, 201, 202, 203, 204, 205, 206, 207});
  sched.submit(make_cached_request("b", other, {5, 13, 25}, 1));
  sched.run_to_completion();
  // The close entry (slot 1) is evicted for b's snapshot; the attach at 12
  // (slot 0) stands.
  const std::string expected_b = expected_a + " F:1 X:0:12@0 P:0:29 N:0:24@1 C:0";
  require(engine.op_stream() == expected_b,
          "the attach target survives:\n  got:      " + engine.op_stream() +
              "\n  expected: " + expected_b);
  require(sched.meters().prefix_hits == 1 && sched.meters().prefix_evictions == 1,
          "one hit, one eviction");
}

DGPP_TEST(scheduler_prefixCache_optOutAndNoArenaKeepThePlainOpStream) {
  const auto run = [](bool arena, bool no_cache) {
    FakeEngine engine(/*slots=*/1, /*total_blocks=*/100, /*block_tokens=*/4);
    if (arena) engine.set_prefix_arena(2, 4);
    engine.arm(0, {1, 2}, 2);
    engine.arm(0, {1, 2}, 2);
    Scheduler sched(&engine, {kEos});
    SchedulerRequest a = make_cached_request("a", counted_prompt(21), {5, 13}, 2);
    SchedulerRequest b = make_cached_request("b", counted_prompt(21), {5, 13}, 2);
    a.no_cache = no_cache;
    b.no_cache = no_cache;
    sched.submit(a);
    sched.submit(b);
    sched.run_to_completion();
    return engine.op_stream();
  };
  const std::string plain = "P:0:21 S:0:1 C:0 P:0:21 S:0:1 C:0";
  require(run(false, false) == plain, "no arena: the plain op stream");
  require(run(true, true) == plain, "opted out: the plain op stream");
  require(run(true, false) != plain, "the cache on changes the stream (the control)");
}

DGPP_TEST(scheduler_prefixCache_evictsTheLeastRecentlyUsedUnattachedEntry) {
  // GIVEN a one-slot arena and three prompts with different first tokens:
  FakeEngine engine(/*slots=*/1, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.set_prefix_arena(/*slots=*/1, /*align=*/4);
  for (int i = 0; i < 4; ++i) engine.arm(0, {1}, 1);
  Scheduler sched(&engine, {kEos});
  sched.submit(make_cached_request("a", counted_prompt(21, 100), {5, 13}, 1));
  sched.submit(make_cached_request("b", counted_prompt(21, 300), {5, 13}, 1));
  sched.submit(make_cached_request("a2", counted_prompt(21, 100), {5, 13}, 1));
  sched.submit(make_cached_request("b2", counted_prompt(21, 300), {5, 13}, 1));
  sched.run_to_completion();
  // THEN each new prompt evicts the other's entry (LRU, unattached — the
  // previous request retired) before its own snapshot; a2 misses (a's
  // entry went), b2 misses (b's entry went).
  const std::string expected =
      "P:0:21 N:0:12@0 C:0 F:0 P:0:21 N:0:12@0 C:0 F:0 P:0:21 N:0:12@0 C:0 "
      "F:0 P:0:21 N:0:12@0 C:0";
  require(engine.op_stream() == expected,
          "eviction:\n  got:      " + engine.op_stream() + "\n  expected: " + expected);
  const Scheduler::Meters m = sched.meters();
  require(m.prefix_evictions == 3 && m.prefix_hits == 0 && m.prefix_misses == 4 &&
              m.prefix_entries == 1,
          "three evictions, no hits, one entry live");
}

DGPP_TEST(scheduler_prefixCache_entriesGiveTheirBlocksToARequestThatNeedsThem) {
  // GIVEN a pool of 12 blocks (48 tokens): a's entry pins 3 blocks; a
  // request needing 10 blocks (37 tokens + 3 steps) cannot fit beside it
  // (12 - 3 = 9 free) — the entry is evicted for it, deterministically.
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/12, /*block_tokens=*/4);
  engine.set_prefix_arena(/*slots=*/2, /*align=*/4);
  engine.arm(0, {1}, 1);
  engine.arm(0, {2, 3, 4}, 3);
  Scheduler sched(&engine, {kEos});
  sched.submit(make_cached_request("a", counted_prompt(21), {5, 13}, 1));
  sched.run_to_completion();
  require(sched.meters().prefix_blocks_pinned == 3, "a's entry pins 3 blocks");
  sched.submit(make_cached_request("big", counted_prompt(37, 500), {}, 3));
  sched.run_to_completion();
  const std::string expected = "P:0:21 N:0:12@0 C:0 F:0 P:0:37 S:0:2 S:0:3 C:0";
  require(engine.op_stream() == expected,
          "block pressure:\n  got:      " + engine.op_stream() + "\n  expected: " + expected);
  require(sched.meters().prefix_evictions == 1 && sched.meters().prefix_blocks_pinned == 0,
          "the entry gave its blocks");
}

DGPP_TEST(scheduler_prefixCache_attachedEntriesAreNeverEvicted) {
  // GIVEN a one-slot arena: a's entry is attached by b (live, cap 4) when
  // c (a different prompt) wants a slot for its own snapshot — none is
  // evictable, so c's snapshot is skipped and counted; b's answer is
  // unaffected. When b itself retires at an aligned position (24), its
  // reference has returned first, so a's entry is evictable for b's own
  // close entry — and is evicted.
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.set_prefix_arena(/*slots=*/1, /*align=*/4);
  engine.arm(0, {1}, 1);
  engine.arm(0, {5, 6, 7, 8}, 4);
  engine.arm(1, {9}, 1);
  Scheduler sched(&engine, {kEos});
  sched.submit(make_cached_request("a", counted_prompt(21), {5, 13}, 1));
  sched.run_to_completion();
  sched.submit(make_cached_request("b", counted_prompt(21), {5, 13}, 4));
  sched.submit(make_cached_request("c", counted_prompt(21, 300), {5, 13}, 1));
  sched.run_to_completion();
  const std::string expected =
      "P:0:21 N:0:12@0 C:0 X:0:12@0 P:0:21 S:0:5 P:1:21 C:1 S:0:6 S:0:7 F:0 RS:0:24@0 C:0";
  require(engine.op_stream() == expected,
          "attached entry:\n  got:      " + engine.op_stream() + "\n  expected: " + expected);
  require(sched.meters().prefix_skipped == 1 && sched.meters().prefix_evictions == 1 &&
              sched.meters().prefix_close_entries == 1,
          "c's snapshot skipped while b was attached; a's entry gave way to b's close entry");
}

// ---- the NVMe cold tier (issue #26) -------------------------------------------
// The commits ride the journal in production; here the driver hands the
// fake's completions straight back to the scheduler before each tick, the
// way a world of one does, until nothing is pending on either side.
void settle(Scheduler& sched, FakeEngine& engine, int max_ticks = 1000) {
  for (int i = 0; i < max_ticks; ++i) {
    std::vector<Scheduler::DiskCommit> commits;
    for (const auto& c : sched.poll_disk_completions()) commits.push_back({c.op, c.ok});
    sched.apply_disk_commits(commits);
    const bool more = sched.tick();
    if (!more && !sched.has_pending() && sched.disk_ops_pending() == 0) return;
  }
  (void)engine;
  throw std::runtime_error("settle: the scheduler did not drain in " + std::to_string(max_ticks) + " ticks");
}

DGPP_TEST(scheduler_nvmeCache_spillsAnEvictedEntryAndRestoresItForALaterRequest) {
  // GIVEN a one-slot arena over a slab of 16 pages, a 21-token prompt
  // whose deepest cut is 12:
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.set_prefix_arena(/*slots=*/1, /*align=*/4);
  engine.set_disk(/*pages=*/16);
  engine.arm(0, {1, 2, 3}, 3);
  engine.arm(0, {4, 5}, 2);
  engine.arm(0, {6, 7, 8}, 3);
  Scheduler sched(&engine, {kEos});
  require(sched.disk_enabled(), "the tier is on");
  // a: a cold prefill takes the entry at 12 and the tier copies it out.
  sched.submit(make_cached_request("a", counted_prompt(21), {5, 13}, 3));
  settle(sched, engine);
  Scheduler::Meters m = sched.meters();
  require(m.disk_spills == 1 && m.disk_spilled == 1 && m.disk_entries == 1 && m.prefix_entries == 1,
          "a's entry spilled and committed; memory keeps it too");
  require(engine.op_stream().find("N:0:12@0") != std::string::npos &&
              engine.op_stream().find("DS:1:0") != std::string::npos,
          "the snapshot, then its spill: " + engine.op_stream());
  // b: another prompt wants the only slot — a's memory entry goes, its
  // cold copy stays.
  sched.submit(make_cached_request("b", counted_prompt(21, 300), {5, 13}, 2));
  settle(sched, engine);
  m = sched.meters();
  require(m.prefix_evictions == 1 && m.disk_entries == 2 && m.disk_spilled == 2,
          "a evicted from memory, b spilled too");
  // c: a's prompt again — no memory entry, a cold copy at its cut: the
  // restore runs, then c attaches at 12 and prefills the suffix.
  sched.submit(make_cached_request("c", counted_prompt(21), {5, 13}, 3));
  settle(sched, engine);
  m = sched.meters();
  require(m.disk_restores == 1 && m.disk_restored == 1 && m.disk_restore_failed == 0,
          "one restore, committed");
  require(m.prefix_hits == 1 && m.prefix_tokens_saved == 12 && m.disk_tokens_restored == 12,
          "c attached to the restored entry at 12");
  const std::string ops = engine.op_stream();
  const size_t dr = ops.find("DR:");
  const size_t attach = ops.find("X:0:12@0");
  require(dr != std::string::npos && attach != std::string::npos && dr < attach,
          "the restore precedes the attach: " + ops);
  require(ids_joined(sched.results()[2].generated) == "6,7,8", "c's ids");
  require(m.disk_pages_used <= m.disk_pages_total, "within capacity");
}

DGPP_TEST(scheduler_nvmeCache_aFailedRestoreIsAColdMissAndDropsTheEntry) {
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.set_prefix_arena(/*slots=*/1, /*align=*/4);
  engine.set_disk(/*pages=*/16);
  engine.arm(0, {1, 2, 3}, 3);
  engine.arm(0, {4, 5}, 2);
  engine.arm(0, {6, 7, 8}, 3);
  Scheduler sched(&engine, {kEos});
  sched.submit(make_cached_request("a", counted_prompt(21), {5, 13}, 3));
  settle(sched, engine);
  sched.submit(make_cached_request("b", counted_prompt(21, 300), {5, 13}, 2));
  settle(sched, engine);
  require(sched.meters().disk_entries == 2, "two cold copies");
  // The restore's read fails (a checksum, a device error): the world's
  // verdict is a failure, c prefills cold, the entry is gone from disk.
  engine.fail_next_disk_op();
  sched.submit(make_cached_request("c", counted_prompt(21), {5, 13}, 3));
  settle(sched, engine);
  const Scheduler::Meters m = sched.meters();
  require(m.disk_restores == 1 && m.disk_restored == 0 && m.disk_restore_failed == 1, "the restore failed");
  require(m.prefix_hits == 0 && m.prefix_misses == 3, "c was a miss");
  require(m.disk_entries == 2 && m.disk_spilled == 3,
          "the suspect entry was dropped; b's stays and c's own cold prefill spilled");
  require(m.prefix_entries == 1, "c's own entry is in memory now");
  require(engine.op_stream().find("P:0:21 ") != std::string::npos ||
              engine.op_stream().rfind("P:0:21") != std::string::npos,
          "c prefilled its whole prompt: " + engine.op_stream());
  require(ids_joined(sched.results()[2].generated) == "6,7,8", "c's ids");
  require(m.pool_blocks_in_use == m.prefix_blocks_pinned, "the failed restore's blocks were released");
}

DGPP_TEST(scheduler_nvmeCache_aFailedSpillLeavesTheMemoryEntry) {
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.set_prefix_arena(/*slots=*/2, /*align=*/4);
  engine.set_disk(/*pages=*/16);
  engine.arm(0, {1, 2, 3}, 3);
  engine.arm(0, {6, 7, 8}, 3);
  Scheduler sched(&engine, {kEos});
  engine.fail_next_disk_op();
  sched.submit(make_cached_request("a", counted_prompt(21), {5, 13}, 3));
  settle(sched, engine);
  Scheduler::Meters m = sched.meters();
  require(m.disk_spills == 1 && m.disk_spilled == 0 && m.disk_spill_failed == 1 && m.disk_entries == 0,
          "the spill failed and left no cold copy");
  require(m.prefix_entries == 1, "the memory entry stands");
  // A repeat hits in memory as before; the entry is not spilled again
  // (nothing asks for it), and nothing is restored.
  sched.submit(make_cached_request("b", counted_prompt(21), {5, 13}, 3));
  settle(sched, engine);
  m = sched.meters();
  require(m.prefix_hits == 1 && m.disk_restores == 0, "a memory hit, no restore");
}

DGPP_TEST(scheduler_nvmeCache_retentionHoldsTheCapacityAndEvictsLru) {
  // A slab of 6 pages with one-page blobs: an entry at 12 tokens takes 3
  // block pages + 1, so one fits; every later spill evicts the oldest.
  FakeEngine engine(/*slots=*/1, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.set_prefix_arena(/*slots=*/1, /*align=*/4);
  engine.set_disk(/*pages=*/6);
  Scheduler sched(&engine, {kEos});
  for (int i = 0; i < 4; ++i) {
    engine.arm(0, {1}, 1);
    sched.submit(make_cached_request(std::to_string(i), counted_prompt(21, 1000 * (i + 1)), {5, 13}, 1));
    settle(sched, engine);
    const Scheduler::Meters m = sched.meters();
    require(m.disk_pages_used <= m.disk_pages_total, "never past the capacity");
    require(m.disk_entries == 1, "one entry fits at a time");
  }
  const Scheduler::Meters m = sched.meters();
  require(m.disk_spilled == 4 && m.disk_evictions == 3, "each spill evicted its predecessor");
  // An entry too large for the slab is skipped, not squeezed in.
  engine.arm(0, {1}, 1);
  sched.submit(make_cached_request("big", counted_prompt(41, 9000), {5, 13, 39}, 1));
  settle(sched, engine);
  require(sched.meters().disk_spill_skipped == 1 && sched.meters().disk_entries == 1,
          "an entry past the capacity is skipped and the slab keeps what it held");
}

DGPP_TEST(scheduler_nvmeCache_twoRequestsWaitOnOneRestore) {
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.set_prefix_arena(/*slots=*/1, /*align=*/4);
  engine.set_disk(/*pages=*/16, /*blob_pages=*/1, /*min_tokens=*/0, /*latency=*/3);
  engine.arm(0, {1, 2, 3}, 3);
  engine.arm(0, {4, 5}, 2);
  engine.arm(0, {6, 7}, 2);
  engine.arm(0, {8, 9}, 2);  // c retires within its admission tick; d takes slot 0 next
  Scheduler sched(&engine, {kEos});
  sched.submit(make_cached_request("a", counted_prompt(21), {5, 13}, 3));
  settle(sched, engine);
  sched.submit(make_cached_request("b", counted_prompt(21, 300), {5, 13}, 2));
  settle(sched, engine);
  // c and d want a's state: one restore serves both.
  sched.submit(make_cached_request("c", counted_prompt(21), {5, 13}, 2));
  sched.submit(make_cached_request("d", counted_prompt(21), {5, 13}, 2));
  settle(sched, engine);
  const Scheduler::Meters m = sched.meters();
  require(m.disk_restores == 1 && m.disk_restored == 1, "one restore");
  require(m.prefix_hits == 2 && m.prefix_tokens_saved == 24, "both attached at 12");
  require(ids_joined(sched.results()[2].generated) == "6,7" && ids_joined(sched.results()[3].generated) == "8,9",
          "both answers");
}

DGPP_TEST(scheduler_nvmeCache_aCancelWhileRestoringKeepsTheRestoredEntry) {
  FakeEngine engine(/*slots=*/2, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.set_prefix_arena(/*slots=*/1, /*align=*/4);
  engine.set_disk(/*pages=*/16, /*blob_pages=*/1, /*min_tokens=*/0, /*latency=*/4);
  engine.arm(0, {1, 2, 3}, 3);
  engine.arm(0, {4, 5}, 2);
  engine.arm(0, {6, 7}, 2);
  Scheduler sched(&engine, {kEos});
  sched.submit(make_cached_request("a", counted_prompt(21), {5, 13}, 3));
  settle(sched, engine);
  sched.submit(make_cached_request("b", counted_prompt(21, 300), {5, 13}, 2));
  settle(sched, engine);
  sched.submit(make_cached_request("c", counted_prompt(21), {5, 13}, 2));
  // The tick starts the restore and does no engine work while it runs
  // (it reports nothing pending for the pass; the request waits).
  (void)sched.tick();
  require(sched.disk_ops_pending() == 1, "c's restore begins at its first tick");
  require(sched.cancel("c"), "c leaves while waiting");
  settle(sched, engine);
  Scheduler::Meters m = sched.meters();
  require(m.disk_restored == 1 && m.prefix_entries == 1, "the restore landed anyway");
  require(sched.results()[2].status == Scheduler::Result::Status::kCancelled, "c cancelled");
  // The next request with a's prompt hits the restored entry in memory.
  sched.submit(make_cached_request("d", counted_prompt(21), {5, 13}, 2));
  settle(sched, engine);
  m = sched.meters();
  require(m.prefix_hits == 1 && m.disk_restores == 1, "a memory hit, no second restore");
}

DGPP_TEST(scheduler_nvmeCache_sameStreamTwice_identicalOpsAndDigests) {
  const auto run = [](bool fail_restore) {
    FakeEngine engine(/*slots=*/2, /*total_blocks=*/100, /*block_tokens=*/4);
    engine.set_prefix_arena(/*slots=*/1, /*align=*/4);
    engine.set_disk(/*pages=*/16, /*blob_pages=*/1, /*min_tokens=*/0, /*latency=*/2);
    engine.arm(0, {1, 2, 3}, 3);
    engine.arm(0, {4, 5}, 2);
    engine.arm(0, {6, 7, 8}, 3);
    Scheduler sched(&engine, {kEos});
    sched.submit(make_cached_request("a", counted_prompt(21), {5, 13}, 3));
    settle(sched, engine);
    sched.submit(make_cached_request("b", counted_prompt(21, 300), {5, 13}, 2));
    settle(sched, engine);
    if (fail_restore) engine.fail_next_disk_op();
    sched.submit(make_cached_request("c", counted_prompt(21), {5, 13}, 3));
    settle(sched, engine);
    return std::make_pair(engine.op_stream(), sched.prefix_digest());
  };
  const auto one = run(false), two = run(false), three = run(true);
  require(one.first == two.first && one.second == two.second, "identical streams: identical ops and digests");
  require(one.second != three.second, "a failed restore: a different digest");
  require(one.first.find("DR:") != std::string::npos, "the base run restored: " + one.first);
}

DGPP_TEST(scheduler_prefixCache_sameStreamTwice_identicalOpsAndDigests) {
  // Rank identity in miniature: two schedulers over two fakes, fed the same
  // requests, produce the same op streams and digests; a third whose last
  // prompt differs (a miss where the others hit) produces another digest.
  const auto run = [](bool variant) {
    FakeEngine engine(/*slots=*/2, /*total_blocks=*/100, /*block_tokens=*/4);
    engine.set_prefix_arena(/*slots=*/3, /*align=*/4);
    engine.arm(0, {1, 2, 3, 4, 5, kEos}, 8);
    engine.arm(1, {6, 7}, 2);
    engine.arm(1, {8}, 1);  // c lands on slot 1, freed first (b's two steps)
    Scheduler sched(&engine, {kEos});
    sched.submit(make_cached_request("a", counted_prompt(21), {5, 13}, 8));
    sched.submit(make_cached_request("b", counted_prompt(21, 300), {5, 13}, 2));
    sched.submit(make_cached_request("c", counted_prompt(21, variant ? 500 : 100),
                                     {5, 13}, 1));
    sched.run_to_completion();
    return std::make_pair(engine.op_stream(), sched.prefix_digest());
  };
  const auto one = run(false), two = run(false), three = run(true);
  require(one.first == two.first && one.second == two.second,
          "identical streams: identical ops and digests");
  require(one.second != three.second, "a different stream: a different digest");
  require(one.first.find("X:1:12@0") != std::string::npos,
          "c attached to a's entry in the base run: " + one.first);
}

DGPP_TEST(scheduler_stop_retiresAtTheNextSweepAsDoneWithReasonStop) {
  // GIVEN a request generating four tokens over a one-slot engine,
  FakeEngine engine(/*slots=*/1, /*total_blocks=*/100, /*block_tokens=*/4);
  engine.arm(0, {10, 11, 12, 13}, /*max_steps=*/4);
  Scheduler sched(&engine, {kEos});
  sched.submit(make_request("a", 5, 4));
  require(sched.tick(), "tick 1: the prefill pick and one step");
  require(sched.meters().decode_steps == 1 && sched.meters().tokens_generated == 2,
          "two tokens after tick 1");
  // WHEN the service's stop lands between ticks (the stop string matched
  // on rank 0; a peer applies the same one from the journal),
  require(sched.stop("a"), "a live request takes the stop");
  require(!sched.stop("nobody"), "an unknown id does not");
  // THEN the next tick's sweep retires it before any engine op: Done with
  // Reason::kStop, two tokens, no further step; a late stop is a no-op.
  require(!sched.tick(), "tick 2: the sweep retires it, nothing remains");
  const Scheduler::Result& res = sched.results()[0];
  require(res.status == Scheduler::Result::Status::kDone &&
              res.reason == Scheduler::Result::Reason::kStop && res.steps_done == 2,
          "Done with Reason::kStop after two tokens");
  require(sched.meters().decode_steps == 1, "no step ran in the retiring tick");
  require(!sched.stop("a"), "a late stop is a no-op");
  // A cancel that arrives with the stop wins (the client is gone).
  FakeEngine engine2(/*slots=*/1, /*total_blocks=*/100, /*block_tokens=*/4);
  engine2.arm(0, {20, 21, 22}, /*max_steps=*/3);
  Scheduler sched2(&engine2, {kEos});
  sched2.submit(make_request("b", 5, 3));
  require(sched2.tick(), "b: tick 1");
  require(sched2.stop("b") && sched2.cancel("b"), "both land");
  require(!sched2.tick(), "b: retired at the sweep");
  require(sched2.results()[0].status == Scheduler::Result::Status::kCancelled &&
              sched2.results()[0].reason == Scheduler::Result::Reason::kCancelled,
          "the cancel wins");
}

DGPP_TEST(scheduler_logitBias_armsTheEngineAfterTheGrammarAndIsRefusedWithoutSupport) {
  // GIVEN a sampling-capable engine and a request carrying a logit_bias,
  FakeEngine engine(/*slots=*/1, /*total_blocks=*/100, /*block_tokens=*/4,
                    /*batch_capacity=*/1, /*can_sample=*/true);
  engine.arm(0, {10, 11}, /*max_steps=*/2);
  Scheduler sched(&engine, {kEos});
  SchedulerRequest r = make_request("a", 5, 2);
  r.logit_bias = {{3, -100.0f}, {7, 1.5f}};
  sched.submit(r);
  sched.run_to_completion();
  // THEN the engine was armed with the two entries before the prefill (the
  // op stream pins the position; a greedy request arms no sampling spec),
  std::string ops;
  for (const std::string& op : engine.ops()) ops += op + " ";
  const size_t b = ops.find("B:0:2 "), p = ops.find("P:0:");
  require(b != std::string::npos && p != std::string::npos && b < p,
          "the bias arming precedes the prefill: " + ops);
  // A greedy-only engine refuses the request at submit; so does an entry
  // with a negative token id.
  FakeEngine greedy(/*slots=*/1, /*total_blocks=*/100, /*block_tokens=*/4);
  Scheduler s2(&greedy, {kEos});
  const auto refused = [](Scheduler& s, const SchedulerRequest& req) {
    try {
      s.submit(req);
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  require(refused(s2, r), "a greedy-only engine refuses a logit_bias");
  SchedulerRequest bad = make_request("c", 5, 2);
  bad.logit_bias = {{-1, 1.0f}};
  require(refused(sched, bad), "a negative token id is refused");
}

int main() {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");
  return dgpp::test::run_all();
}

DGPP_TEST(scheduler_images_bypass_token_cache_and_grouping) {
  class ImageEngine : public GroupFakeEngine {
   public:
    ImageEngine() : GroupFakeEngine(2, 100, 4, 32, 64) {}
    int calls = 0;
    bool supports_images() const override { return true; }
    int32_t prefill_images(int req, const std::vector<int64_t>& p,
                           const std::vector<dgpp::ImageInput>& images) override {
      require(images.size() == 1 && images[0].rgb[0] == 123, "image reaches prefill intact");
      ++calls;
      return FakeEngine::prefill(req, p);
    }
  } engine;
  engine.set_prefix_arena(4, 1, 2);
  engine.arm(0, {10}, 1);
  engine.arm(0, {11}, 1);
  Scheduler sched(&engine, {});
  SchedulerRequest r;
  r.id = "image-one";
  r.prompt = {1, 2, 3, 4, 5, 6};
  r.images.push_back({1, 1, 28, 28, std::vector<uint8_t>(28 * 28 * 3, 123)});
  sched.submit(r);
  r.id = "image-two";
  sched.submit(r);
  while (sched.tick()) {
  }
  require(engine.calls == 2, "both images take their own prefill");
  require(engine.op_stream().find("PG:") == std::string::npos,
          "images excluded from grouped text prefill");
  require(engine.op_stream().find("N:") == std::string::npos &&
              engine.op_stream().find("X:") == std::string::npos &&
              engine.op_stream().find("RS:") == std::string::npos,
          "no image state stored or attached in token-only cache");
  FakeEngine text(1, 100, 4);
  Scheduler text_sched(&text, {});
  bool threw = false;
  try {
    text_sched.submit(r);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "text engine cannot silently discard image data");
}

DGPP_TEST(prefix_cache_image_identity_and_partial_prefixes) {
  using Cache = dgpp::sched::PrefixCache;
  Cache cache({8, 1, 4});
  const std::vector<int64_t> ids(24, 154854);
  std::vector<dgpp::ImageInput> images{{8, 2, 28, 56, std::vector<uint8_t>(28 * 56 * 3, 123)}};
  auto red = cache.image_keys(images);
  const auto hash = [&](int64_t n, const Cache::Images& keys) {
    return Cache::with_images(Cache::hash_prefix(ids.data(), n), n, keys);
  };
  const int before = cache.insert(ids.data(), 4, cache.take_free_slot(), 1, red);
  const int at = cache.insert(ids.data(), 8, cache.take_free_slot(), 2, red);
  const int inside = cache.insert(ids.data(), 9, cache.take_free_slot(), 3, red);
  const int after = cache.insert(ids.data(), 16, cache.take_free_slot(), 4, red);
  auto repeated = cache.image_keys(images);
  require(repeated[0].input == red[0].input, "repeated pixels share immutable storage");
  require(cache.lookup(ids, {4, 8, 9, 16}, {hash(4, red), hash(8, red), hash(9, red), hash(16, red)},
                       repeated) == after, "identical image attaches after its span");
  require(cache.find_exact(ids.data(), 4, hash(4, {})) == before, "text before image is reusable");
  require(cache.find_exact(ids.data(), 8, hash(8, {})) < 0, "MTP lookahead at image start is keyed");
  require(cache.find_exact(ids.data(), 9, hash(9, repeated), repeated) == inside,
          "a snapshot inside an image retains its whole identity");
  require(cache.find_exact(ids.data(), 8, hash(8, repeated), repeated) == at, "image-start hit");
  images[0].rgb.back() ^= 1;
  auto changed = cache.image_keys(images);
  require(cache.lookup(ids, {4, 8, 9, 16}, {hash(4, changed), hash(8, changed), hash(9, changed), hash(16, changed)},
                       changed) == before, "changed pixels reuse only the earlier text prefix");
  // Deliberately collide the image hashes: byte comparison must still refuse.
  changed[0].hash = red[0].hash;
  require(cache.find_exact(ids.data(), 16, hash(16, changed), changed) < 0,
          "image hash collisions cannot produce a cache hit");
  images[0].rgb.back() ^= 1;
  std::swap(images[0].width, images[0].height);
  auto reshaped = cache.image_keys(images);
  require(cache.find_exact(ids.data(), 16, hash(16, reshaped), reshaped) < 0,
          "identical RGB bytes with changed geometry must miss");
  images = {*red[0].input};
  images.push_back({20, 1, 28, 28, std::vector<uint8_t>(28 * 28 * 3, 45)});
  auto appended = cache.image_keys(images);
  require(cache.find_exact(ids.data(), 16, hash(16, appended), appended) == after,
          "a new suffix image preserves the cached prefix");
  images[0].offset = 7;
  auto moved = cache.image_keys(images);
  require(cache.find_exact(ids.data(), 16, hash(16, moved), moved) < 0,
          "image position participates in identity");
  require(cache.nearest(ids, moved).common == 7, "miss diagnostics account for image changes");
  const int occupied = cache.take_free_slot();
  require(cache.insert(ids.data(), 16, occupied, 5, repeated) < 0, "same image entry deduplicates");
  cache.give_back_slot(occupied);
  for (int i = 0; i < 4; ++i) require(cache.evict_lru() >= 0, "image entries remain evictable");
  require(cache.live_entries() == 0 && red[0].input->rgb.back() == 123,
          "eviction releases entries without invalidating a request's image key");
}

DGPP_TEST(scheduler_images_reuse_prefill_and_generated_continuations) {
  class CachedImages : public FakeEngine {
   public:
    CachedImages() : FakeEngine(1, 1000, 4) {}
    bool supports_images() const override { return true; }
    bool supports_image_prefix_cache() const override { return true; }
    std::vector<int64_t> attaches;
    int32_t prefill_cached(int req, const std::vector<int64_t>& prompt, PrefixPrefill* plan) override {
      require(plan->images && plan->images->size() == 1, "cached prefill receives image embeddings' inputs");
      attaches.push_back(plan->attach_position);
      return FakeEngine::prefill_cached(req, prompt, plan);
    }
  } engine;
  engine.set_prefix_arena(12, 4, 2048);
  for (int i = 0; i < 4; ++i) engine.arm(0, {30, 31, 32, 33}, 4);
  Scheduler sched(&engine, {});
  auto first = make_cached_request("image-first", counted_prompt(21), {5, 13}, 4);
  first.images.push_back({2, 1, 28, 28, std::vector<uint8_t>(28 * 28 * 3, 123)});
  sched.submit(first);
  while (sched.tick()) {}
  auto next = first;
  next.id = "image-continuation";
  next.prompt.insert(next.prompt.end(), {30, 31, 32, 33, 40, 41});
  next.boundaries.push_back(25);
  sched.submit(next);
  while (sched.tick()) {}
  require(engine.attaches == std::vector<int64_t>({0, 24}),
          "generated continuation survives retirement and attaches on the next turn");
  next.id = "image-changed";
  next.images[0].rgb[0] = 45;
  sched.submit(next);
  while (sched.tick()) {}
  require(engine.attaches.back() == 0, "same token ids with different pixels cannot attach");
  first.id = "image-repeated";
  sched.submit(first);
  while (sched.tick()) {}
  require(engine.attaches.back() == 12, "the earlier image's snapshot remains reusable");
}

DGPP_TEST(prefix_cache_image_byte_budget_shares_identity_and_preserves_entries) {
  using Cache = dgpp::sched::PrefixCache;
  Cache::Config config;
  config.slots = 4;
  config.image_bytes = 28 * 28 * 3;
  Cache cache(config);
  std::vector<int64_t> tokens(12, 7);
  std::vector<dgpp::ImageInput> images{{1, 1, 28, 28, std::vector<uint8_t>(config.image_bytes, 123)}};
  const auto first = cache.image_keys(images);
  const int a = cache.insert(tokens.data(), 4, cache.take_free_slot(), 1, first);
  require(a >= 0, "first image fits byte budget");
  cache.attach(a, 2);
  const auto repeat = cache.image_keys(images);
  require(first[0].input == repeat[0].input, "shared image identity");
  const int b = cache.insert(tokens.data(), 8, cache.take_free_slot(), 3, repeat);
  require(b >= 0 && cache.image_bytes() == config.image_bytes, "second prefix does not double count pixels");
  images[0].rgb[0] ^= 1;
  const auto changed = cache.image_keys(images);
  const int spare = cache.take_free_slot();
  require(cache.insert(tokens.data(), 12, spare, 4, changed) < 0, "new pixels exceed the byte budget");
  cache.give_back_slot(spare);
  require(cache.entry(a).live && cache.entry(b).live && cache.entry(a).attached == 1,
          "image storage pressure preserves existing and attached prefixes");
  require(cache.stats().skipped_image_bytes == 1, "image storage pressure is observable");
  require(cache.evict_lru() >= 0 && cache.image_bytes() == config.image_bytes,
          "shared pixels stay while an entry references them");
  cache.detach(a);
  require(cache.evict_lru() >= 0 && cache.image_bytes() == 0, "last eviction releases image identity");
}
