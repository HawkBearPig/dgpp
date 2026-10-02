#include "sched/scheduler.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <stdexcept>
#include <string>
#include <utility>

#include "common/log.hpp"

namespace dgpp::sched {

std::vector<std::vector<int32_t>> SchedulerEngine::step_batch(
    const std::vector<int>& reqs) {
  if (reqs.empty())
    throw std::invalid_argument("SchedulerEngine: empty decode batch");
  if (reqs.size() > static_cast<size_t>(decode_batch_capacity()))
    throw std::invalid_argument(
        "SchedulerEngine: decode batch exceeds advertised capacity");
  std::vector<std::vector<int32_t>> out;
  out.reserve(reqs.size());
  for (const int req : reqs) out.push_back(step(req));
  return out;
}

namespace {

// A request's lifetime token footprint: the prompt plus every token it
// can generate. The last generated token is never fed back, so a one-row
// decode writes no position past prompt + max_steps - 2 and this count
// carries one token of slack; a two-row speculative verify (the MTP graph
// engine, which steps only while fewer than max_steps tokens stand) writes
// its second row at prompt + max_steps - 1 at the latest, so the slack is
// exactly its second row. This count is also what SchedulerEngine::reserve
// pins in the engine, so it must not shrink.
int64_t reserve_tokens(const SchedulerRequest& r) {
  return static_cast<int64_t>(r.prompt.size()) + r.max_steps;
}

const char* reason_name(Scheduler::Result::Reason r) {
  switch (r) {
    case Scheduler::Result::Reason::kEos: return "eos";
    case Scheduler::Result::Reason::kSteps: return "steps";
    case Scheduler::Result::Reason::kCancelled: return "cancelled";
    case Scheduler::Result::Reason::kPoolExhausted: return "pool exhausted";
    default: return "none";
  }
}

}  // namespace

Scheduler::Scheduler(SchedulerEngine* engine,
                     std::vector<int64_t> eos_token_ids, int queue_limit,
                     AdmissionPolicy policy, int prefix_slots)
    : engine_(engine),
      eos_ids_(std::move(eos_token_ids)),
      queue_limit_(queue_limit),
      policy_(policy) {
  if (engine_ == nullptr)
    throw std::invalid_argument("Scheduler: engine must not be null");
  // The prefix cache (M7): as many slots as asked, never more than the
  // engine's arena — every rank must resolve the same count (the warm
  // record carries rank 0's), and a short arena on one rank is a
  // configuration error, not a quiet degradation.
  prefix_info_ = engine_->prefix_info();
  if (prefix_slots < -1)
    throw std::invalid_argument("Scheduler: prefix_slots must be -1, 0 or positive");
  const int slots = prefix_slots < 0 ? prefix_info_.arena_slots : prefix_slots;
  if (slots > prefix_info_.arena_slots)
    throw std::invalid_argument(
        "Scheduler: prefix cache asks for " + std::to_string(slots) +
        " snapshot slots but the engine's arena holds " +
        std::to_string(prefix_info_.arena_slots));
  PrefixCache::Config pc;
  pc.slots = slots;
  pc.align = std::max<int64_t>(1, prefix_info_.align);
  pc.chunk_tokens = std::max<int64_t>(1, prefix_info_.chunk_tokens);
  cache_ = PrefixCache(pc);
  if (policy_.window_tokens < 1)
    throw std::invalid_argument(
        "Scheduler: the admission window must be at least one token");
  if (policy_.prefill_budget_tokens < 0)
    throw std::invalid_argument("Scheduler: prefill budget must be nonnegative");
  if (policy_.prefill_budget_tokens > 0) {
    const int64_t align = engine_->prefill_chunk_alignment();
    if (align < 1 || policy_.prefill_budget_tokens % align != 0 ||
        policy_.prefill_budget_tokens > engine_->prefill_chunk_limit())
      throw std::invalid_argument("Scheduler: prefill budget needs a supported engine, aligned chunks and "
                                  "a budget no larger than the engine's prefill limit");
  }
  if (policy_.prefill_idle_budget_tokens < 0 ||
      (policy_.prefill_idle_budget_tokens > 0 &&
       (policy_.prefill_budget_tokens == 0 ||
        policy_.prefill_idle_budget_tokens < policy_.prefill_budget_tokens ||
        policy_.prefill_idle_budget_tokens % engine_->prefill_chunk_alignment() != 0 ||
        policy_.prefill_idle_budget_tokens > engine_->prefill_chunk_limit())))
    throw std::invalid_argument("Scheduler: idle prefill budget needs an enabled budget and must be aligned, "
                                "at least the busy budget and no larger than the engine's prefill limit");
  if (policy_.prefix_min_tokens < 0)
    throw std::invalid_argument("Scheduler: prefix_min_tokens must be nonnegative");
  if (queue_limit_ < 0)
    throw std::invalid_argument(
        "Scheduler: queue_limit must be 0 (unbounded) or positive");
  slots_.assign(static_cast<size_t>(engine_->max_concurrent_requests()), -1);
  if (slots_.empty())
    throw std::invalid_argument(
        "Scheduler: engine reports zero concurrent request slots");
  decode_batch_capacity_ = engine_->decode_batch_capacity();
  if (decode_batch_capacity_ < 1 ||
      decode_batch_capacity_ > static_cast<int>(slots_.size()))
    throw std::invalid_argument(
        "Scheduler: engine decode batch capacity must be in [1, " +
        std::to_string(slots_.size()) + "]");
}

bool Scheduler::is_eos(int32_t token) const {
  // Three ids at real dims — a linear scan beats building a set.
  return std::find(eos_ids_.begin(), eos_ids_.end(),
                   static_cast<int64_t>(token)) != eos_ids_.end();
}

int64_t Scheduler::initial_reserve_tokens(const SchedulerRequest& spec) const {
  const int64_t full = reserve_tokens(spec);
  if (policy_.mode != AdmissionPolicy::Mode::kGrowOnDemand) return full;
  // The window, never less than one step's width plus the pending token
  // (a step writes at most max_tokens_per_step rows past the KV's end).
  const int64_t headroom =
      std::max<int64_t>(policy_.window_tokens, engine_->max_tokens_per_step() + 1);
  return std::min<int64_t>(full, static_cast<int64_t>(spec.prompt.size()) + headroom);
}

int64_t Scheduler::reserve_blocks(const Request& r) const {
  return engine_->blocks_for_tokens(initial_reserve_tokens(r.spec));
}

int Scheduler::youngest_active_after(int arrival) const {
  for (int i = static_cast<int>(requests_.size()) - 1; i > arrival; --i)
    if (requests_[static_cast<size_t>(i)].state == State::kActive ||
        requests_[static_cast<size_t>(i)].state == State::kPrefilling) return i;
  return -1;
}

void Scheduler::grow_reservations() {
  if (policy_.mode != AdmissionPolicy::Mode::kGrowOnDemand) return;
  const int64_t width = engine_->max_tokens_per_step();
  for (size_t i = 0; i < requests_.size(); ++i) {  // arrival order = priority
    Request& r = requests_[i];
    if (r.state != State::kActive) continue;
    const int64_t prompt = static_cast<int64_t>(r.spec.prompt.size());
    const int64_t full = prompt + r.spec.max_steps;
    // The KV holds prompt + steps_done - 1 tokens (the last pick is pending);
    // the next step writes up to `width` more. One token of slack.
    const int64_t need = std::min<int64_t>(full, prompt + r.steps_done + width);
    if (need <= r.reserved_tokens) continue;
    int64_t target = std::min<int64_t>(
        full, std::max<int64_t>(need, r.reserved_tokens + policy_.window_tokens));
    const int64_t held = engine_->blocks_for_tokens(r.reserved_tokens);
    for (;;) {
      const int64_t free_blocks =
          engine_->pool_blocks_total() - engine_->pool_blocks_in_use();
      if (engine_->blocks_for_tokens(target) - held <= free_blocks) break;
      if (engine_->blocks_for_tokens(need) - held <= free_blocks) {
        target = need;  // the minimum, rather than shedding for a full window
        break;
      }
      // Even the minimum does not fit: the youngest reserved request goes —
      // this one when nothing younger is live.
      const int victim = youngest_active_after(static_cast<int>(i));
      const int shed = victim >= 0 ? victim : static_cast<int>(i);
      Request& v = requests_[static_cast<size_t>(shed)];
      DGPP_LOG_WARN(
          "sched: pool exhausted — request '{}' shed after {} tokens so "
          "request '{}' can write its next step ({} blocks free of {}; "
          "finish_reason length)",
          v.spec.id, v.steps_done, r.spec.id, free_blocks,
          engine_->pool_blocks_total());
      ++pool_sheds_;
      retire(shed, Result::Status::kDone, Result::Reason::kPoolExhausted);
      if (shed == static_cast<int>(i)) break;
    }
    if (r.state != State::kActive) continue;
    engine_->reserve(r.slot, target);
    r.reserved_tokens = target;
    ++grows_;
    if (observer_) observer_->on_grow(r.spec.id, target);
    DGPP_LOG_INFO(
        "sched: request '{}' reservation grown to {} tokens ({} blocks; pool "
        "{}/{} blocks in use)",
        r.spec.id, target, engine_->blocks_for_tokens(target),
        engine_->pool_blocks_in_use(), engine_->pool_blocks_total());
  }
}

int Scheduler::free_slot() const {
  for (int s = 0; s < static_cast<int>(slots_.size()); ++s)
    if (slots_[static_cast<size_t>(s)] < 0) return s;
  return -1;
}

Scheduler::PrefixPlan Scheduler::plan_prefix(const Request& r) const {
  PrefixPlan plan;
  if (!cache_on(r)) return plan;
  const int e = cache_.lookup(r.spec.prompt, r.cuts, r.cut_hashes, r.cache_images);
  if (e >= 0) {
    plan.attach_entry = e;
    plan.attach_position = cache_.entry(e).position;
  }
  // No entry below the policy's floor: a short prompt attaches to what
  // exists but never takes a slot (the single-Spark replay of 2026-09-28:
  // three 45-token probes per turn pushed a 66K conversation out of a
  // 13-slot arena every turn; with the floor the same arena served it).
  const int64_t floor_tokens = policy_.prefix_min_tokens;
  // A new entry at the deepest cut past the attach — the next turn's cut.
  if (!r.cuts.empty() && r.cuts.back() > plan.attach_position &&
      r.cuts.back() >= floor_tokens)
    plan.snap_position = r.cuts.back();
  // Keep the deepest cut for identical repeats and one regular chunk cut
  // at least a chunk before the end for a shared document with a new tail.
  const int64_t n = static_cast<int64_t>(r.spec.prompt.size());
  const int64_t chunk = cache_.config().chunk_tokens;
  const int64_t align = cache_.config().align;
  if (prefix_info_.body_snapshots && n / chunk >= 4) {
    const int64_t body = (n / chunk - 1) * chunk;
    if (body > plan.attach_position && body < plan.snap_position && body % align == 0 &&
        body >= floor_tokens)
      plan.body_snap_position = body;
  }
  // The head cut (2026-09-28): the aligned image of the prompt's first
  // structural boundary past its start — a long system prompt's end — so
  // the next conversation under the same system prompt, or the turn after
  // a client compacted its history, attaches there instead of prefilling
  // the head cold (the replay: 18K tokens on each of 24 post-compaction
  // turns). Taken on the same walk as the deepest cut, under the same
  // floor, only on a family whose extra snapshots preserve the walk.
  if (policy_.prefix_head_snapshots && prefix_info_.body_snapshots) {
    for (const int64_t b : r.spec.boundaries) {
      const int64_t head = (b / align) * align;
      if (head <= 0) continue;  // the opening marker: no prefix before it
      if (head >= floor_tokens && head > plan.attach_position && head < plan.snap_position &&
          head != plan.body_snap_position)
        plan.head_snap_position = head;
      break;
    }
  }
  return plan;
}

std::vector<int64_t> Scheduler::cut_boundaries(const Request& r) const {
  std::vector<int64_t> out;
  if (!cache_on(r)) return out;
  const int64_t align = cache_.config().align;
  for (const int64_t b : r.spec.boundaries) {
    const int64_t image = align > 0 ? (b / align) * align : b;
    if (image >= policy_.prefix_min_tokens) out.push_back(b);
  }
  return out;
}

int64_t Scheduler::snapshot_blocks(int64_t position) const {
  const int64_t bt = prefix_info_.block_tokens;
  return bt > 0 && position > 0 && position % bt != 0 ? 1 : 0;
}

bool Scheduler::snapshot_needs_block(int64_t position, int64_t previous) const {
  return snapshot_blocks(position) > 0 &&
         (previous < 0 || snapshot_blocks(previous) == 0);
}

bool Scheduler::ensure_free_blocks(int64_t need, const std::string& id) {
  if (need <= 0) return true;
  for (;;) {
    const int64_t free =
        engine_->pool_blocks_total() - engine_->pool_blocks_in_use();
    if (free >= need) return true;
    const int slot = cache_.evict_lru();
    if (slot < 0) return false;
    free_arena_slot(slot);
    emit_prefix(id, "evict", 0, slot);
  }
}

int64_t Scheduler::new_blocks(const Request& r, const PrefixPlan& plan) const {
  int64_t blocks = reserve_blocks(r);
  const int64_t bt = prefix_info_.block_tokens;
  if (plan.attach_entry >= 0 && bt > 0)
    blocks -= plan.attach_position / bt;  // the full blocks come shared
  // The pool blocks the cache takes on top of the reservation (the soak's
  // sweep of 2026-09-05 found the admission one block short of them): the
  // cut entry's private partial-block copy, and one block of headroom for
  // the request's own rolling or hop snapshot's copy — a position aligned
  // to kpool but not to the block, which exists only when the block is
  // wider than the alignment.
  if (plan.snap_position > 0) blocks += snapshot_blocks(plan.snap_position);
  if (plan.body_snap_position > 0) blocks += snapshot_blocks(plan.body_snap_position);
  if (plan.head_snap_position > 0) blocks += snapshot_blocks(plan.head_snap_position);
  if (cache_on(r) && bt > std::max<int64_t>(1, prefix_info_.align)) blocks += 1;
  return std::max<int64_t>(blocks, 0);
}

int Scheduler::next_admissible() {
  int64_t free_blocks =
      engine_->pool_blocks_total() - engine_->pool_blocks_in_use();
  for (size_t i = 0; i < requests_.size(); ++i) {
    if (requests_[i].state != State::kQueued) continue;
    // No head-of-line blocking: the OLDEST request that FITS admits. A
    // large deferred request must not dam the queue behind it — the
    // starvation it could suffer under an unbounded small-request stream
    // is a Stage 4 bounded-queue problem, not a policy bug.
    if (free_slot() < 0) return -1;
    // A pool too small for the request WITH the cache's blocks (the cut
    // entry's copy, the rolling headroom) but large enough without them
    // admits the request cache-less rather than never: the same decision on
    // every rank, from the same journaled request against the same pool.
    if (cache_on(requests_[i]) &&
        new_blocks(requests_[i], plan_prefix(requests_[i])) >
            engine_->pool_blocks_total() &&
        reserve_blocks(requests_[i]) <= engine_->pool_blocks_total()) {
      requests_[i].cache_off = true;
      DGPP_LOG_INFO(
          "sched: request '{}' runs without the prefix cache — the pool "
          "({} blocks) cannot hold its reservation and the cache's blocks",
          requests_[i].spec.id, engine_->pool_blocks_total());
    }
    if (new_blocks(requests_[i], plan_prefix(requests_[i])) <= free_blocks)
      return static_cast<int>(i);
    // The prefix cache's entries pin blocks; the LRU unattached ones give
    // way to a request that needs them (never a block a live request
    // holds). The planned entry is touched first so it is evicted last —
    // and the plan is recomputed after every eviction, because the victim
    // may have been that entry.
    if (cache_.enabled()) {
      bool evicted = false;
      for (;;) {
        const PrefixPlan plan = plan_prefix(requests_[i]);
        if (new_blocks(requests_[i], plan) <= free_blocks) break;
        if (plan.attach_entry >= 0) cache_.touch(plan.attach_entry, ticks_);
        const int slot = cache_.evict_lru();
        if (slot < 0) break;
        free_arena_slot(slot);
        emit_prefix(requests_[i].spec.id, "evict", 0, slot);
        evicted = true;
        free_blocks = engine_->pool_blocks_total() - engine_->pool_blocks_in_use();
      }
      if (evicted &&
          new_blocks(requests_[i], plan_prefix(requests_[i])) <= free_blocks)
        return static_cast<int>(i);
    }
  }
  return -1;
}

void Scheduler::validate_new(const SchedulerRequest& request) const {
  if (request.id.empty())
    throw std::invalid_argument("Scheduler: request id must not be empty");
  for (const Request& r : requests_) {
    if (r.spec.id == request.id)
      throw std::invalid_argument("Scheduler: duplicate request id '" +
                                  request.id + "'");
  }
  if (request.prompt.empty())
    throw std::invalid_argument("Scheduler: request '" + request.id +
                               "' has an empty prompt");
  validate_image_inputs(request.images, request.prompt.size());
  if (!request.images.empty() && !engine_->supports_images())
    throw std::invalid_argument("Scheduler: this engine does not support image inputs");
  if (request.max_steps < 1)
    throw std::invalid_argument("Scheduler: request '" + request.id +
                                "' must generate at least one token");
  try {
    sample::validate_params(request.sampling);
  } catch (const std::invalid_argument& e) {
    throw std::invalid_argument("Scheduler: request '" + request.id +
                                "' has an invalid sampling spec: " + e.what());
  }
  if (request.sampling.temperature > 0.0f && !engine_->supports_sampling())
    throw std::invalid_argument(
        "Scheduler: request '" + request.id +
        "' asks for stochastic sampling but the engine is greedy-only");
  if (request.logprobs >= 0 && !engine_->supports_logprobs())
    throw std::invalid_argument(
        "Scheduler: request '" + request.id +
        "' asks for logprobs but the engine reports none");
  if (request.logprobs >= 0 && request.sampling.logprobs != request.logprobs)
    throw std::invalid_argument(
        "Scheduler: request '" + request.id +
        "' logprobs and sampling.logprobs disagree");
  if (request.grammar.active() && !engine_->supports_constraints())
    throw std::invalid_argument(
        "Scheduler: request '" + request.id +
        "' asks for constrained decoding but the engine cannot mask the "
        "pick");
  if (!request.logit_bias.empty() && !engine_->supports_logit_bias())
    throw std::invalid_argument(
        "Scheduler: request '" + request.id +
        "' carries a logit_bias but the engine cannot bias the pick");
  for (const LogitBias& b : request.logit_bias)
    if (b.token < 0 || !std::isfinite(b.bias))
      throw std::invalid_argument("Scheduler: request '" + request.id +
                                  "' has an invalid logit_bias entry");
  if (request.cancel_after < 0 || request.cancel_after > request.max_steps)
    throw std::invalid_argument(
        "Scheduler: request '" + request.id + "' cancel_after must be in "
        "[0, max_steps] — a cancel that can never fire is a manifest error");
  int64_t last = 0;
  for (const int64_t b : request.boundaries) {
    if (b <= last || b >= static_cast<int64_t>(request.prompt.size()))
      throw std::invalid_argument(
          "Scheduler: request '" + request.id +
          "' boundaries must be ascending positions inside the prompt");
    last = b;
  }
}

int Scheduler::queued_count() const {
  int n = 0;
  for (const Request& r : requests_)
    if (r.state == State::kQueued) ++n;
  return n;
}

bool Scheduler::try_submit(SchedulerRequest request) {
  validate_new(request);
  if (queue_limit_ > 0 && queued_count() >= queue_limit_) return false;
  Request r;
  r.spec = std::move(request);
  if (cache_on(r)) {
    r.cache_images = cache_.image_keys(r.spec.images);
    // The prompt's cuts and their prefix hashes, once: the lookups at every
    // tick this request waits are then O(cuts) probes.
    r.cuts = cache_.cuts(static_cast<int64_t>(r.spec.prompt.size()),
                         r.spec.boundaries);
    r.cut_hashes.reserve(r.cuts.size());
    uint64_t h = PrefixCache::kSeed;
    int64_t done = 0;
    for (const int64_t c : r.cuts) {
      for (; done < c; ++done) h = PrefixCache::extend_hash(h, r.spec.prompt[static_cast<size_t>(done)]);
      r.cut_hashes.push_back(PrefixCache::with_images(h, c, r.cache_images));
    }
  }
  requests_.push_back(std::move(r));
  results_.emplace_back();
  return true;
}

void Scheduler::submit(SchedulerRequest request) {
  if (!try_submit(std::move(request)))
    throw QueueFullError(
        "Scheduler: admission queue full (" +
        std::to_string(queued_count()) + " queued, limit " +
        std::to_string(queue_limit_) + ") — shed load or raise the limit");
}

bool Scheduler::cancel(const std::string& id) {
  for (Request& r : requests_) {
    if (r.spec.id != id) continue;
    if (r.state != State::kTerminal) {
      r.cancel_requested = true;
      return true;
    }
    return false;  // terminal: a late cancel is a no-op, never an error
  }
  return false;
}

bool Scheduler::stop(const std::string& id) {
  for (Request& r : requests_) {
    if (r.spec.id != id) continue;
    if (r.state != State::kTerminal) {
      r.stop_requested = true;
      return true;
    }
    return false;  // terminal: a late stop is a no-op
  }
  return false;
}

bool Scheduler::has_pending() const {
  for (const Request& r : requests_)
    if (r.state != State::kTerminal) return true;
  return false;
}

const Scheduler::Result* Scheduler::find(const std::string& id) const {
  for (size_t i = 0; i < requests_.size(); ++i)
    if (requests_[i].spec.id == id) return &results_[i];
  return nullptr;
}

int Scheduler::admit_prepare(int arrival) {
  Request& r = requests_[static_cast<size_t>(arrival)];
  const int slot = free_slot();
  if (slot < 0)
    throw std::logic_error("Scheduler: admit without a free slot");
  // The spec lands on the slot before its first pick (the prefill's); the
  // grammar with it, so the prefill pick is the first constrained position.
  engine_->configure_sampling(slot, r.spec.sampling, r.spec.seed);
  engine_->configure_logprobs(slot, r.spec.logprobs);
  engine_->configure_constraint(slot, r.spec.grammar);
  engine_->configure_logit_bias(slot, r.spec.logit_bias);
  // The slot is taken for the group's other members' free_slot() scans.
  slots_[static_cast<size_t>(slot)] = arrival;
  engine_->prefill_monitor()->begin(slot, r.spec.id, static_cast<int64_t>(r.spec.prompt.size()));
  if (observer_) observer_->on_admit(r.spec.id, slot);
  DGPP_LOG_INFO("sched: request '{}' starting prefill in slot {} ({} prompt tokens)",
                r.spec.id, slot, r.spec.prompt.size());
  return slot;
}

std::vector<int> Scheduler::admissible_group(int first, int64_t budget) {
  std::vector<int> group;
  const int64_t span_limit = engine_->prefill_group_span_limit();
  const int64_t total_limit = budget > 0
      ? std::min<int64_t>(engine_->prefill_group_total_limit(), budget)
      : engine_->prefill_group_total_limit();
  if (span_limit <= 0 || total_limit <= 0) return group;
  const auto groupable = [&](const Request& r) {
    if (r.state != State::kQueued || !r.spec.images.empty()) return false;
    const int64_t P = static_cast<int64_t>(r.spec.prompt.size());
    if (P <= 0 || P > span_limit) return false;
    if (!cache_on(r)) return true;
    const PrefixPlan plan = plan_prefix(r);
    return plan.attach_entry < 0 && plan.snap_position <= 0;
  };
  if (!groupable(requests_[static_cast<size_t>(first)])) return group;
  group.push_back(first);
  int64_t total = static_cast<int64_t>(requests_[static_cast<size_t>(first)].spec.prompt.size());
  int64_t free_blocks = engine_->pool_blocks_total() - engine_->pool_blocks_in_use() -
                        reserve_blocks(requests_[static_cast<size_t>(first)]);
  int open_slots = static_cast<int>(std::count(slots_.begin(), slots_.end(), -1)) - 1;
  for (size_t i = static_cast<size_t>(first) + 1; i < requests_.size() && open_slots > 0; ++i) {
    const Request& r = requests_[i];
    if (!groupable(r)) continue;
    const int64_t P = static_cast<int64_t>(r.spec.prompt.size());
    if (total + P > total_limit) continue;
    const int64_t need = reserve_blocks(r);
    if (need > free_blocks) continue;
    group.push_back(static_cast<int>(i));
    total += P;
    free_blocks -= need;
    --open_slots;
  }
  return group;
}

void Scheduler::admit_group(const std::vector<int>& arrivals) {
  if (arrivals.size() < 2) throw std::logic_error("Scheduler: a group admission of fewer than two");
  std::vector<int> slots;
  std::vector<const std::vector<int64_t>*> prompts;
  for (const int arrival : arrivals) {
    slots.push_back(admit_prepare(arrival));
    prompts.push_back(&requests_[static_cast<size_t>(arrival)].spec.prompt);
  }
  const auto t_prefill = std::chrono::steady_clock::now();
  std::vector<int32_t> tokens;
  try {
    tokens = engine_->prefill_group(slots, prompts);
  } catch (...) {
    for (const int slot : slots) {
      slots_[static_cast<size_t>(slot)] = -1;
      engine_->prefill_monitor()->finish(slot);
    }
    throw;
  }
  const double prefill_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t_prefill)
                                .count();
  if (tokens.size() != arrivals.size())
    throw std::runtime_error("Scheduler: the engine's group prefill returned " + std::to_string(tokens.size()) +
                             " tokens for " + std::to_string(arrivals.size()) + " requests");
  prefill_ms_ += prefill_ms;  // one physical group, not one execution per member
  for (size_t i = 0; i < arrivals.size(); ++i) {
    Request& r = requests_[static_cast<size_t>(arrivals[i])];
    if (cache_on(r)) {
      ++cache_.stats().misses;
      log_prefix_miss(r);
    }
    // The group's wall on every member: each waited for the whole forward.
    admit_finish(arrivals[i], slots[i], tokens[i], prefill_ms, 0);
  }
  DGPP_LOG_INFO("sched: {} requests admitted together in one prefill ({:.0f} ms)", arrivals.size(), prefill_ms);
}

void Scheduler::admit(int arrival) {
  Request& r = requests_[static_cast<size_t>(arrival)];
  const int slot = admit_prepare(arrival);
  int32_t token = -1;
  int64_t attached = 0;  // prompt tokens an attach skipped (meters)
  const auto t_prefill = std::chrono::steady_clock::now();
  if (!cache_on(r)) {
    // No cache for this request: the pre-cache op, exactly.
    try {
      token = r.spec.images.empty() ? engine_->prefill(slot, r.spec.prompt)
                                    : engine_->prefill_images(slot, r.spec.prompt, r.spec.images);
    } catch (...) {
      engine_->prefill_monitor()->finish(slot);
      slots_[static_cast<size_t>(slot)] = -1;
      throw;
    }
  } else {
    // The prefix cache's plan (M7): attach to the deepest matching entry at
    // one of the prompt's cuts, and save the deepest cut plus an earlier
    // document cut for long prompts. The deepest cut gets an arena slot
    // first; a full arena may skip the extra snapshot.
    const PrefixPlan plan = plan_prefix(r);
    const std::vector<int64_t> cuts = cut_boundaries(r);
    SchedulerEngine::PrefixPrefill pp;
    pp.boundaries = &cuts;
    pp.images = &r.spec.images;
    int snap_slot = -1;
    if (plan.attach_entry >= 0) {
      // Attach first: an attached entry is never evicted, and the snapshot
      // slot acquired next may evict the LRU unattached entry — which, with
      // the arena full, was this very entry (the curve sweep's find of
      // 2026-09-05: "PrefixArena: slot 25 is empty" — the victim's slot came
      // back as the snapshot slot, and the prefill attached to nothing).
      cache_.attach(plan.attach_entry, ticks_);
      pp.attach_slot = cache_.entry(plan.attach_entry).slot;
      pp.attach_position = plan.attach_position;
    }
    if (plan.snap_position > 0) {
      snap_slot = acquire_arena_slot(r.spec.id);
      if (snap_slot >= 0) {
        pp.snap_slot = snap_slot;
        pp.snap_position = plan.snap_position;
      } else {
        ++cache_.stats().skipped_no_slot;
      }
    }
    // The head cut before the body cut: shared by every conversation under
    // the system prompt, it is the more valuable entry when slots are scarce.
    if (plan.head_snap_position > 0) {
      pp.head_snap_slot = acquire_arena_slot(r.spec.id);
      if (pp.head_snap_slot >= 0)
        pp.head_snap_position = plan.head_snap_position;
      else
        ++cache_.stats().skipped_no_slot;
    }
    if (plan.body_snap_position > 0) {
      pp.body_snap_slot = acquire_arena_slot(r.spec.id);
      if (pp.body_snap_slot >= 0)
        pp.body_snap_position = plan.body_snap_position;
      else
        ++cache_.stats().skipped_no_slot;
    }
    try {
      engine_->prefill_monitor()->begin(slot, r.spec.id, static_cast<int64_t>(r.spec.prompt.size()),
                                       pp.attach_position);
      token = engine_->prefill_cached(slot, r.spec.prompt, &pp);
    } catch (...) {
      engine_->prefill_monitor()->finish(slot);
      slots_[static_cast<size_t>(slot)] = -1;
      if (plan.attach_entry >= 0) cache_.detach(plan.attach_entry);
      if (snap_slot >= 0) free_arena_slot(snap_slot);
      if (pp.head_snap_slot >= 0) free_arena_slot(pp.head_snap_slot);
      if (pp.body_snap_slot >= 0) free_arena_slot(pp.body_snap_slot);
      throw;
    }
    if (plan.attach_entry >= 0) {
      r.attach_entry = plan.attach_entry;
      r.attach_position = plan.attach_position;
      attached = plan.attach_position;
      emit_prefix(r.spec.id, "attach", plan.attach_position, pp.attach_slot);
    } else {
      ++cache_.stats().misses;
      log_prefix_miss(r);
    }
    finish_prefill_snapshot(r, snap_slot, plan.snap_position, pp.snap_taken);
    finish_prefill_snapshot(r, pp.head_snap_slot, pp.head_snap_position, pp.head_snap_taken,
                            /*head=*/true);
    finish_prefill_snapshot(r, pp.body_snap_slot, pp.body_snap_position, pp.body_snap_taken);
  }
  const double prefill_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t_prefill)
                                .count();
  prefill_ms_ += prefill_ms;
  admit_finish(arrival, slot, token, prefill_ms, attached);
}

void Scheduler::begin_prefill(int arrival, int64_t budget) {
  Request& r = requests_[static_cast<size_t>(arrival)];
  const int slot = admit_prepare(arrival);
  const std::vector<int64_t> cuts = cut_boundaries(r);
  SchedulerEngine::PrefixPrefill pp;
  pp.boundaries = &cuts;
  pp.images = &r.spec.images;
  r.admitted_at = std::chrono::steady_clock::now();
  try {
    if (cache_on(r)) {
      const PrefixPlan plan = plan_prefix(r);
      if (plan.attach_entry >= 0) {
        cache_.attach(plan.attach_entry, ticks_);
        r.attach_entry = plan.attach_entry;
        r.attach_position = plan.attach_position;
        pp.attach_slot = cache_.entry(plan.attach_entry).slot;
        pp.attach_position = plan.attach_position;
        emit_prefix(r.spec.id, "attach", plan.attach_position, pp.attach_slot);
      } else {
        ++cache_.stats().misses;
        log_prefix_miss(r);
      }
      if (plan.snap_position > 0) {
        r.prefill_snap_slot = acquire_arena_slot(r.spec.id);
        r.prefill_snap_position = plan.snap_position;
        if (r.prefill_snap_slot >= 0) {
          pp.snap_slot = r.prefill_snap_slot;
          pp.snap_position = plan.snap_position;
        } else ++cache_.stats().skipped_no_slot;
      }
      if (plan.head_snap_position > 0) {
        r.prefill_head_slot = acquire_arena_slot(r.spec.id);
        r.prefill_head_position = plan.head_snap_position;
        if (r.prefill_head_slot >= 0) {
          pp.head_snap_slot = r.prefill_head_slot;
          pp.head_snap_position = plan.head_snap_position;
        } else
          ++cache_.stats().skipped_no_slot;
      }
      if (plan.body_snap_position > 0) {
        r.prefill_body_slot = acquire_arena_slot(r.spec.id);
        r.prefill_body_position = plan.body_snap_position;
        if (r.prefill_body_slot >= 0) {
          pp.body_snap_slot = r.prefill_body_slot;
          pp.body_snap_position = plan.body_snap_position;
        } else
          ++cache_.stats().skipped_no_slot;
      }
    }
    const int64_t reserved = initial_reserve_tokens(r.spec);
    engine_->prefill_monitor()->begin(slot, r.spec.id, static_cast<int64_t>(r.spec.prompt.size()),
                                     pp.attach_position);
    engine_->begin_prefill(slot, r.spec.prompt, reserved, budget, pp);
    r.reserved_tokens = reserved;
  } catch (...) {
    engine_->prefill_monitor()->finish(slot);
    slots_[static_cast<size_t>(slot)] = -1;
    if (r.attach_entry >= 0) {
      cache_.detach(r.attach_entry);
      r.attach_entry = -1;
    }
    if (r.prefill_snap_slot >= 0) {
      free_arena_slot(r.prefill_snap_slot);
      r.prefill_snap_slot = -1;
    }
    if (r.prefill_body_slot >= 0) {
      free_arena_slot(r.prefill_body_slot);
      r.prefill_body_slot = -1;
    }
    throw;
  }
  r.prefill_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - r.admitted_at).count();
  prefill_ms_ += r.prefill_ms;
  r.attached_tokens = pp.attach_position;
  prompt_tokens_ += r.attached_tokens;
  r.admitted = true;
  r.slot = slot;
  r.state = State::kPrefilling;
  slots_[static_cast<size_t>(slot)] = arrival;
  DGPP_LOG_INFO("sched: request '{}' prefilling in slot {} ({} tokens/tick)",
                r.spec.id, slot, budget);
}

void Scheduler::advance_prefill(int arrival, int64_t budget) {
  Request& r = requests_[static_cast<size_t>(arrival)];
  const auto started = std::chrono::steady_clock::now();
  const auto progress = engine_->advance_prefill(r.slot, budget);
  const double ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
  prefill_ms_ += ms;
  r.prefill_ms += ms;
  if (progress.computed_tokens <= 0 || progress.computed_tokens > budget)
    throw std::runtime_error("Scheduler: prefill chunk made no progress or exceeded its token budget");
  prefill_cursor_ = arrival;
  r.prefill_computed += progress.computed_tokens;
  engine_->prefill_monitor()->update(r.slot, r.attached_tokens + r.prefill_computed);
  prompt_tokens_ += progress.computed_tokens;
  prompt_tokens_computed_ += progress.computed_tokens;
  const int64_t expected = static_cast<int64_t>(r.spec.prompt.size()) - r.attached_tokens;
  if (r.prefill_computed > expected || progress.first_token < -1 ||
      (r.prefill_computed == expected && progress.first_token < 0))
    throw std::runtime_error("Scheduler: invalid prefill completion progress");
  if (progress.first_token < 0) return;
  if (r.prefill_computed + r.attached_tokens != static_cast<int64_t>(r.spec.prompt.size()))
    throw std::runtime_error("Scheduler: prefill completed at the wrong prompt position");
  finish_prefill_snapshot(r, r.prefill_snap_slot, r.prefill_snap_position, progress.snap_taken);
  r.prefill_snap_slot = -1;
  finish_prefill_snapshot(r, r.prefill_head_slot, r.prefill_head_position,
                          progress.head_snap_taken, /*head=*/true);
  r.prefill_head_slot = -1;
  finish_prefill_snapshot(r, r.prefill_body_slot, r.prefill_body_position,
                          progress.body_snap_taken);
  r.prefill_body_slot = -1;
  admit_finish(arrival, r.slot, progress.first_token, r.prefill_ms, r.attached_tokens, true);
}

bool Scheduler::needs_chunked_prefill(int arrival, int64_t budget) const {
  const Request& r = requests_[static_cast<size_t>(arrival)];
  return budget > 0 &&
         (r.spec.images.empty() || engine_->supports_image_chunked_prefill()) &&
         static_cast<int64_t>(r.spec.prompt.size()) > budget;
}

bool Scheduler::admit_fitting(int64_t& tick_cap, int64_t budget, bool first_prefill) {
  // Zero is the monolithic sentinel to admissible_group(), not a remaining
  // budget. Stop before either grouping or the prefix-cache eviction scan.
  if (budget > 0 && tick_cap <= 0) return false;
  const int first = next_admissible();
  if (first < 0) return false;
  const int64_t P =
      static_cast<int64_t>(requests_[static_cast<size_t>(first)].spec.prompt.size());
  if (needs_chunked_prefill(first, budget)) {
    // A chunked read-in consumes the tick: it begins (and takes its first
    // chunk) only when the caller allows new chunked starts, and always
    // ends the admit loop — chunked read-ins stay one at a time.
    if (!first_prefill) return false;
    begin_prefill(first, budget);
    advance_prefill(first, budget);
    tick_cap = 0;
    return true;
  }
  // The prospective fit: a fitting one-shot past the tick's remaining cap
  // waits for a later tick, so one tick never stacks unbounded synchronous
  // prefill work. (With a zero budget there is no cap: monolithic mode
  // admits exactly as before, and the caller ends the loop after one.)
  // Engines without image chunking retain their monolithic fallback. An
  // oversized image must be the tick's only prefill admission, even when
  // it finishes immediately and releases its slot.
  const bool monolithic_image = !requests_[static_cast<size_t>(first)].spec.images.empty() &&
                                !engine_->supports_image_chunked_prefill();
  if (budget > 0 && P > tick_cap && !(first_prefill && monolithic_image)) return false;
  const std::vector<int> group =
      admissible_group(first, budget > 0 ? std::min(budget, tick_cap) : budget);
  if (group.size() >= 2) {
    int64_t total = 0;
    for (const int a : group)
      total += static_cast<int64_t>(requests_[static_cast<size_t>(a)].spec.prompt.size());
    admit_group(group);
    tick_cap -= total;
    return true;
  }
  admit(first);
  tick_cap = std::max<int64_t>(0, tick_cap - P);
  return true;
}

void Scheduler::finish_prefill_snapshot(Request& r, int slot, int64_t position, bool taken,
                                        bool head) {
  if (slot < 0) return;
  if (!taken) {
    cache_.give_back_slot(slot);
    return;
  }
  const int64_t next_token =
      prefix_info_.prefill_lookahead ? r.spec.prompt.at(static_cast<size_t>(position)) : -1;
  const int entry =
      cache_.insert(r.spec.prompt.data(), position, slot, ticks_, r.cache_images, next_token);
  if (entry < 0)
    free_arena_slot(slot);
  else {
    ++cache_.stats().snapshots;
    if (head) ++cache_.stats().head_snapshots;
    emit_prefix(r.spec.id, "snapshot", position, slot);
  }
}

void Scheduler::admit_finish(int arrival, int slot, int32_t token, double prefill_ms, int64_t attached,
                              bool resumed) {
  engine_->prefill_monitor()->finish(slot);
  Request& r = requests_[static_cast<size_t>(arrival)];
  const int64_t reserve = reserve_blocks(r);
  const auto t_prefill = std::chrono::steady_clock::now();
  prefill_request_ms_ += resumed
      ? std::chrono::duration<double, std::milli>(t_prefill - r.admitted_at).count() : prefill_ms;
  r.admitted = true;
  if (!resumed) r.admitted_at = t_prefill;
  r.prefill_ms = prefill_ms;
  r.attached_tokens = attached;
  ++prompts_prefilled_;
  if (!resumed) {
    prompt_tokens_ += static_cast<int64_t>(r.spec.prompt.size());
    prompt_tokens_computed_ += static_cast<int64_t>(r.spec.prompt.size()) - attached;
  }
  if (token < 0) {
    slots_[static_cast<size_t>(slot)] = -1;
    engine_->close(slot);
    throw std::runtime_error("Scheduler: engine prefill returned token " +
                             std::to_string(token) + " for request '" +
                             r.spec.id + "'");
  }
  // Capture/device-position decode may not grow the DSA table during a
  // replay. Admission has already proved this reservation fits, and no
  // other scheduler mutation can interleave between that proof and here.
  const int64_t reserved = initial_reserve_tokens(r.spec);
  try {
    engine_->reserve(slot, reserved);
  } catch (...) {
    slots_[static_cast<size_t>(slot)] = -1;
    engine_->close(slot);
    throw;
  }
  r.reserved_tokens = reserved;
  r.state = State::kActive;
  r.slot = slot;
  slots_[static_cast<size_t>(slot)] = arrival;
  if (arrival == deferred_logged_) deferred_logged_ = -1;
  DGPP_LOG_INFO(
      "sched: request '{}' admitted to slot {} (reserve {} blocks; pool "
      "{}/{} blocks in use) — first token {}",
      r.spec.id, slot, reserve, engine_->pool_blocks_in_use(),
      engine_->pool_blocks_total(), token);
  const std::vector<sample::Result> lps = collect_logprobs(arrival, slot, 1);
  (void)append_token(arrival, token, lps.empty() ? nullptr : &lps[0]);
}

std::vector<sample::Result> Scheduler::collect_logprobs(int arrival,
                                                            int slot,
                                                            size_t tokens) {
  const Request& r = requests_[static_cast<size_t>(arrival)];
  if (r.spec.logprobs < 0) return {};
  std::vector<sample::Result> lps = engine_->take_logprobs(slot);
  if (lps.size() != tokens)
    throw std::runtime_error(
        "Scheduler: engine reported " + std::to_string(lps.size()) +
        " logprob entries for " + std::to_string(tokens) +
        " tokens of request '" + r.spec.id + "'");
  return lps;
}

void Scheduler::step_batch(const std::vector<int>& arrivals) {
  if (arrivals.empty())
    throw std::logic_error("Scheduler: empty decode batch");
  std::vector<int> slots;
  slots.reserve(arrivals.size());
  for (const int arrival : arrivals) {
    const Request& r = requests_[static_cast<size_t>(arrival)];
    if (r.state != State::kActive || r.slot < 0)
      throw std::logic_error("Scheduler: decode batch contains an inactive "
                             "request");
    slots.push_back(r.slot);
  }

  // Validate the outer shape before publishing any token. A malformed
  // engine result must not leave half a physical pass visible to clients.
  const auto t_step = std::chrono::steady_clock::now();
  const std::vector<std::vector<int32_t>> batches =
      engine_->step_batch(slots);
  step_ms_ += std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - t_step)
                  .count();
  ++decode_steps_;
  decode_rows_ += static_cast<int64_t>(slots.size());
  for (const int arrival : arrivals)
    ++requests_[static_cast<size_t>(arrival)].decode_passes;
  if (batches.size() != arrivals.size())
    throw std::runtime_error(
        "Scheduler: engine returned " + std::to_string(batches.size()) +
        " request results for a decode batch of " +
        std::to_string(arrivals.size()));
  for (size_t i = 0; i < arrivals.size(); ++i) {
    const Request& r = requests_[static_cast<size_t>(arrivals[i])];
    if (batches[i].empty())
      throw std::runtime_error("Scheduler: engine step returned no tokens for "
                               "request '" + r.spec.id + "'");
    for (const int32_t token : batches[i])
      if (token < 0)
        throw std::runtime_error("Scheduler: engine returned token " +
                                 std::to_string(token) + " for request '" +
                                 r.spec.id + "'");
  }

  cursor_ = arrivals.back();
  // The prefix cache's hops (M7): an armed request whose step committed two
  // tokens had its state at the armed position taken by the engine inside
  // the step — recorded before the tokens are applied, so a retire in this
  // pass finds the rolling slot at the position the close entry wants. A
  // one-token step landed ON the position: the next tick's rolling
  // snapshot (or the retire-time one) takes it.
  for (size_t i = 0; i < arrivals.size(); ++i) {
    Request& r = requests_[static_cast<size_t>(arrivals[i])];
    if (r.hop_armed < 0) continue;
    const int64_t hop = r.hop_armed;
    r.hop_armed = -1;
    if (batches[i].size() < 2) continue;
    r.rolling_position = hop;
    ++cache_.stats().rolling;
    ++cache_.stats().hops;
    cache_.note(4, static_cast<uint64_t>(hop), static_cast<uint64_t>(r.rolling_slot));
    emit_prefix(r.spec.id, "hop", hop, r.rolling_slot);
  }
  for (size_t i = 0; i < arrivals.size(); ++i) {
    const int arrival = arrivals[i];
    const std::vector<int32_t>& tokens = batches[i];
    const std::vector<sample::Result> lps =
        collect_logprobs(arrival, slots[i], tokens.size());
    for (size_t t = 0; t < tokens.size(); ++t) {
      // A speculative pass can have advanced farther than the public request
      // survives. EOS/cap/cancel retires the slot and deliberately drops the
      // rest of this request's batch; its extra device state is never seen.
      // Other requests in the same physical pass remain independent and are
      // still published below.
      requests_[static_cast<size_t>(arrival)].step_tail = static_cast<int>(tokens.size() - 1 - t);
      if (append_token(arrival, tokens[t], lps.empty() ? nullptr : &lps[t]))
        break;
    }
  }
}

bool Scheduler::append_token(int arrival, int32_t token,
                             const sample::Result* logprobs) {
  Request& r = requests_[static_cast<size_t>(arrival)];
  if (token < 0)
    throw std::runtime_error("Scheduler: engine returned token " +
                             std::to_string(token) + " for request '" +
                             r.spec.id + "'");
  ++r.steps_done;
  r.generated.push_back(static_cast<int64_t>(token));
  ++tokens_generated_;
  if (observer_) {
    observer_->on_token(r.spec.id, token, r.steps_done);
    if (logprobs != nullptr)
      observer_->on_token_logprobs(r.spec.id, r.steps_done, *logprobs);
  }
  // Per token — DEBUG since the throughput line (2026-09-06); serve_pace.py
  // reads it under DGPP_LOG_LEVEL=debug.
  DGPP_LOG_DEBUG("sched: request '{}' step {}: token {}", r.spec.id,
                 r.steps_done, token);
  if (is_eos(token) && !r.spec.ignore_eos) {
    retire(arrival, Result::Status::kDone, Result::Reason::kEos);
  } else if (r.spec.cancel_after > 0 &&
             r.steps_done >= r.spec.cancel_after) {
    retire(arrival, Result::Status::kCancelled, Result::Reason::kCancelled);
  } else if (r.steps_done >= r.spec.max_steps) {
    retire(arrival, Result::Status::kDone, Result::Reason::kSteps);
  }
  return r.state == State::kTerminal;
}

void Scheduler::retire(int arrival, Result::Status status,
                       Result::Reason reason) {
  Request& r = requests_[static_cast<size_t>(arrival)];
  if (r.slot >= 0) engine_->prefill_monitor()->finish(r.slot);
  if (r.state == State::kPrefilling)
    prefill_request_ms_ += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - r.admitted_at).count();
  // The slot's draft acceptance for the retire line, read before close
  // resets it.
  const SchedulerEngine::MtpAcceptance acceptance =
      r.slot >= 0 ? engine_->mtp_acceptance(r.slot)
                  : SchedulerEngine::MtpAcceptance{};
  // The prefix cache (M7): the request's attach reference returns first,
  // so the entry it opened from is evictable for its own close entry.
  if (r.attach_entry >= 0) {
    cache_.detach(r.attach_entry);
    r.attach_entry = -1;
  }
  if (r.prefill_snap_slot >= 0) {
    free_arena_slot(r.prefill_snap_slot);
    r.prefill_snap_slot = -1;
  }
  if (r.prefill_body_slot >= 0) {
    free_arena_slot(r.prefill_body_slot);
    r.prefill_body_slot = -1;
  }
  if (r.prefill_head_slot >= 0) {
    free_arena_slot(r.prefill_head_slot);
    r.prefill_head_slot = -1;
  }
  // The prefix cache's retire-time snapshot (M7): when the answer completed
  // with its committed position aligned — the exact position the close
  // entry wants — take the state now, from the live slot, whether or not
  // the rolling slot lags (the MTP graph's two-token steps can hop over
  // every aligned position once the committed count is odd, so the
  // rolling slot alone may sit a pool or more behind).
  // A step cut short (the request completed on an earlier token of a
  // multi-token pass) leaves the model's state past the committed position
  // — the live snapshot cannot be taken there; the rolling entry stands.
  if (r.slot >= 0 && cache_on(r) && r.step_tail == 0 &&
      (reason == Result::Reason::kEos || reason == Result::Reason::kSteps)) {
    const int64_t align = std::max<int64_t>(1, prefix_info_.align);
    const int64_t committed =
        static_cast<int64_t>(r.spec.prompt.size()) + r.steps_done - 1;
    if (committed > 0 && committed % align == 0 && committed != r.rolling_position &&
        committed > r.attach_position && committed >= policy_.prefix_min_tokens) {
      if (r.rolling_slot < 0) r.rolling_slot = acquire_arena_slot(r.spec.id);
      if (r.rolling_slot < 0) {
        ++cache_.stats().skipped_no_slot;
      } else if (snapshot_needs_block(committed, r.rolling_position) &&
                 !ensure_free_blocks(1, r.spec.id)) {
        ++cache_.stats().skipped_no_block;
      } else {
        engine_->prefix_snapshot(r.slot, r.rolling_slot, committed);
        r.rolling_position = committed;
        ++cache_.stats().rolling;
        cache_.note(4, static_cast<uint64_t>(committed),
                    static_cast<uint64_t>(r.rolling_slot));
        emit_prefix(r.spec.id, "rolling", committed, r.rolling_slot);
      }
    }
  }
  // An externally cancelled QUEUED request never held a slot or blocks.
  if (r.slot >= 0) {
    engine_->close(r.slot);
    slots_[static_cast<size_t>(r.slot)] = -1;
  }
  // The prefix cache (M7): the rolling snapshot — the state at the aligned
  // image of the last token's position, where the next turn's cold prefill
  // cuts — becomes an entry when the answer completed (EOS or the cap); a
  // cancelled or shed request's partial answer is not worth a slot.
  if (r.rolling_slot >= 0) {
    const int slot = r.rolling_slot;
    const int64_t position = r.rolling_position;
    r.rolling_slot = -1;
    r.rolling_position = -1;
    bool kept = false;
    if (reason == Result::Reason::kEos || reason == Result::Reason::kSteps) {
      std::vector<int64_t> ids(r.spec.prompt.begin(), r.spec.prompt.end());
      const int64_t gen = position - static_cast<int64_t>(r.spec.prompt.size());
      if (gen > 0)
        ids.insert(ids.end(), r.generated.begin(), r.generated.begin() + gen);
      if (static_cast<int64_t>(ids.size()) == position) {
        const int e = cache_.insert(ids.data(), position, slot, ticks_, r.cache_images);
        if (e >= 0) {
          kept = true;
          ++cache_.stats().close_entries;
          emit_prefix(r.spec.id, "close", position, slot);
        }
      }
    }
    if (!kept) {
      free_arena_slot(slot);
      emit_prefix(r.spec.id, "drop", position, slot);
    }
  }
  r.state = State::kTerminal;
  Result& res = results_[static_cast<size_t>(arrival)];
  res.status = status;
  res.reason = reason;
  res.slot = r.slot;
  res.steps_done = r.steps_done;
  res.generated = std::move(r.generated);
  r.slot = -1;
  ++retired_;
  if (observer_) observer_->on_retire(r.spec.id, res);
  // The request's own numbers: its prefill (the prompt, the
  // tokens an attach skipped, the wall from admission), then its decode —
  // the tokens after the prefill pick, the passes it rode (each shared
  // with every other live request, so ms/pass is the pace this request
  // saw, not the engine's step), tok/s and ms/tok, and MTP's tok/pass.
  std::string line = std::format("sched: request '{}' retired ({}): {} tok",
                                 r.spec.id, reason_name(reason), r.steps_done);
  if (!r.admitted) {
    line += " — never admitted";
  } else {
    const double total_s = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - r.admitted_at)
                               .count();
    const double decode_s = std::max(0.0, total_s - r.prefill_ms / 1000.0);
    const int decode_tokens = std::max(0, r.steps_done - 1);
    line += std::format(
        " in {:.1f} s — prefill {} tok ({} cached) in {:.0f} ms; decode {} tok "
        "/ {} pass{} in {:.1f} s",
        total_s, r.spec.prompt.size(), r.attached_tokens, r.prefill_ms,
        decode_tokens, r.decode_passes, r.decode_passes == 1 ? "" : "es",
        decode_s);
    if (decode_tokens > 0 && r.decode_passes > 0 && decode_s > 0.0)
      line += std::format(
          ": {:.1f} tok/s, {:.1f} ms/tok, {:.0f} ms/pass, {:.2f} tok/pass",
          decode_tokens / decode_s, 1000.0 * decode_s / decode_tokens,
          1000.0 * decode_s / r.decode_passes,
          static_cast<double>(decode_tokens) / r.decode_passes);
    std::string accept;
    for (int p = 0; p < acceptance.depth && p < 8; ++p)
      if (acceptance.attempts[p] > 0)
        accept += std::format(" p{} {:.0f} %", p + 1,
                              100.0 * static_cast<double>(acceptance.accepts[p]) /
                                  static_cast<double>(acceptance.attempts[p]));
    if (!accept.empty()) line += ", accept" + accept;
  }
  DGPP_LOG_INFO("{}", line);
  release_retired(r, res);
}

void Scheduler::release_retired(Request& r, Result& res) {
  // The tombstone keeps the id (the duplicate check, a late cancel's
  // no-op), the state and the counts; the payloads go — swapped with
  // empties so their capacity returns to the allocator, not just their
  // size. Every read of a retired record elsewhere is of the tombstone.
  std::vector<int64_t>().swap(r.spec.prompt);
  std::vector<ImageInput>().swap(r.spec.images);
  PrefixCache::Images().swap(r.cache_images);
  std::vector<int64_t>().swap(r.spec.boundaries);
  std::vector<LogitBias>().swap(r.spec.logit_bias);
  r.spec.grammar = text::GrammarSpec{};
  std::vector<int64_t>().swap(r.cuts);
  std::vector<uint64_t>().swap(r.cut_hashes);
  std::vector<int64_t>().swap(r.generated);
  if (!keep_retired_) std::vector<int64_t>().swap(res.generated);
}

bool Scheduler::tick() {
  const bool more = quantum();
  compact_retired();
  return more;
}

void Scheduler::compact_retired() {
  if (keep_retired_) return;
  size_t live = 0;
  for (const Request& r : requests_)
    if (r.state != State::kTerminal) ++live;
  if (live == requests_.size()) return;
  // Fresh vectors sized to the live requests, so a burst's capacity goes
  // with its records. The slice order is unchanged: the live requests keep
  // their cyclic order, and the cursor moves to the nearest live record at
  // or before it (none: -1), so the next slice starts at the same request.
  std::vector<Request> kept;
  std::vector<Result> kept_results;
  kept.reserve(live);
  kept_results.reserve(live);
  std::vector<int> index(requests_.size(), -1);  // old arrival -> new
  int cursor = -1;
  int prefill_cursor = -1;
  int deferred = -1;
  for (size_t i = 0; i < requests_.size(); ++i) {
    if (requests_[i].state == State::kTerminal) continue;
    index[i] = static_cast<int>(kept.size());
    kept.push_back(std::move(requests_[i]));
    kept_results.push_back(std::move(results_[i]));
    if (static_cast<int>(i) <= cursor_) cursor = index[i];
    if (static_cast<int>(i) <= prefill_cursor_) prefill_cursor = index[i];
    if (static_cast<int>(i) == deferred_logged_) deferred = index[i];
  }
  for (int& arrival : slots_) {
    if (arrival < 0) continue;
    if (index[static_cast<size_t>(arrival)] < 0)
      throw std::logic_error(
          "Scheduler: an engine slot maps to a retired request");
    arrival = index[static_cast<size_t>(arrival)];
  }
  requests_.swap(kept);
  results_.swap(kept_results);
  cursor_ = cursor;
  prefill_cursor_ = prefill_cursor;
  deferred_logged_ = deferred;
}

bool Scheduler::quantum() {
  ++ticks_;
  // The external-cancel sweep — FIXED POSITION, before any admission
  // or step: a request cancelled BETWEEN ticks never pays another
  // engine op; one cancelled MID-TICK waits out the in-flight op (the
  // price of a single fixed position the fabric journal can stamp —
  // every rank retires the same request at the same quantum, and
  // arrival order keeps it deterministic). Observed live at w1: a flag
  // that arrived during a 5-minute prefill rode out the whole tick.
  for (size_t i = 0; i < requests_.size(); ++i) {
    Request& r = requests_[i];
    if (r.state == State::kTerminal) continue;
    if (r.cancel_requested)
      retire(static_cast<int>(i), Result::Status::kCancelled,
             Result::Reason::kCancelled);
    else if (r.stop_requested)  // the stop string's retire
      retire(static_cast<int>(i), Result::Status::kDone, Result::Reason::kStop);
  }
  // Grow-on-demand's fixed position: after the sweep, before any
  // admission or step — the step below never writes past a reservation,
  // and every rank grows or sheds the same requests at the same quantum.
  grow_reservations();

  const bool any_active = std::any_of(
      requests_.begin(), requests_.end(),
      [](const Request& r) { return r.state == State::kActive; });
  const bool any_queued = std::any_of(
      requests_.begin(), requests_.end(),
      [](const Request& r) { return r.state == State::kQueued; });
  const auto prefill = policy_.prefill_budget_tokens > 0
      ? std::find_if(requests_.begin(), requests_.end(), [](const Request& r) { return r.state == State::kPrefilling; })
      : requests_.end();
  const bool prefill_in_flight = prefill != requests_.end();
  if (!any_active && !any_queued && !prefill_in_flight) return false;

  bool progressed = false;
  bool admitted_any = false;  // anything left the queue this tick

  // (1) Admission before the step, so a queued request's first token is not
  // delayed behind a step — and mid-answer requests never wait behind more
  // than one read-in. Every branch below depends only on replicated
  // scheduler state, so every rank computes the same loop.
  // This choice depends only on replicated scheduler state. Reevaluate at
  // every yield so an unfinished prompt speeds up when its decoding peer retires.
  const auto prefill_budget = [&]() -> int64_t {
    return !any_active && policy_.prefill_idle_budget_tokens > 0
        ? policy_.prefill_idle_budget_tokens : policy_.prefill_budget_tokens;
  };
  const int64_t budget = prefill_budget();
  if (prefill_in_flight) {
    // A chunked prefill is in flight: advance a fair slice on equal aligned
    // shares of the tick's budget (all prefills when they fit), begin at most
    // one new chunked read-in, and admit fitting one-shots/groups into the
    // align-down leftover. Order derives from arrival order only, so every
    // rank agrees.
    std::vector<int> inflight;
    for (size_t i = 0; i < requests_.size(); ++i)
      if (requests_[i].state == State::kPrefilling) inflight.push_back(static_cast<int>(i));
    const int64_t align = engine_->prefill_chunk_alignment();
    const size_t max_advances = static_cast<size_t>(budget / align);
    // The one new begin, when any: the oldest fitting chunked-needing
    // request (skip-fit, no eviction dance — a begin must not disturb the
    // pool the in-flight prefills hold), gated so every share keeps at
    // least one aligned chunk.
    int begin_arrival = -1;
    if (free_slot() >= 0 && inflight.size() < max_advances) {
      const int64_t free_blocks =
          engine_->pool_blocks_total() - engine_->pool_blocks_in_use();
      for (size_t i = 0; i < requests_.size(); ++i) {
        if (requests_[i].state != State::kQueued) continue;
        if (!needs_chunked_prefill(static_cast<int>(i), budget)) continue;
        if (new_blocks(requests_[i], plan_prefix(requests_[i])) <= free_blocks) {
          begin_arrival = static_cast<int>(i);
          break;
        }
      }
    }
    if (begin_arrival >= 0) inflight.push_back(begin_arrival);
    if (inflight.size() > max_advances) {
      // Idle admissions may outnumber the smaller busy budget's chunks.
      // Rotate a bounded slice so every unfinished request gets a turn;
      // rounding every share up would silently exceed the tick's cap.
      const auto next = std::upper_bound(inflight.begin(), inflight.end(), prefill_cursor_);
      std::rotate(inflight.begin(), next, inflight.end());
      inflight.resize(max_advances);
    }
    const int64_t n = static_cast<int64_t>(inflight.size());
    // The constructor guarantees budget >= align > 0, so the slice is
    // nonempty and every share is a supported chunk within the total cap.
    const int64_t share = (budget / n / align) * align;
    if (begin_arrival >= 0) {
      begin_prefill(begin_arrival, share);
      admitted_any = true;
    }
    for (const int a : inflight) advance_prefill(a, share);
    progressed = true;
    // The align-down leftover still admits fitting one-shots/groups — but
    // never a new chunked start (chunked read-ins stay one at a time, and
    // the begin above already took this tick's).
    int64_t leftover = budget - n * share;
    while (admit_fitting(leftover, budget, /*first_prefill=*/false)) {
      admitted_any = true;
      progressed = true;
    }
  } else {
    // No prefill in flight: admit fitting queued requests in a loop before
    // the decode slice is built. The loop stops at the tick's prefill
    // budget, at a chunked start, or when nothing fitting remains.
    // Monolithic mode retains exactly one admission event.
    int64_t tick_cap = budget > 0 ? budget : engine_->prefill_chunk_limit();
    while (admit_fitting(tick_cap, budget, /*first_prefill=*/!admitted_any)) {
      admitted_any = true;
      progressed = true;
      // Monolithic mode keeps its original policy bit-for-bit: one
      // admission event per tick. Positive budgets stop at an exhausted
      // cap, including after a chunked start or an oversized image.
      if (budget == 0) break;
    }
  }
  if (!admitted_any && any_queued) {
    // Deferral bookkeeping: log the head of the queue once per
    // deferral episode, with the numbers an operator needs.
    const auto head = std::find_if(
        requests_.begin(), requests_.end(),
        [](const Request& r) { return r.state == State::kQueued; });
    const int head_arrival =
        static_cast<int>(head - requests_.begin());
    if (deferred_logged_ != head_arrival) {
      deferred_logged_ = head_arrival;
      const int64_t free_blocks = engine_->pool_blocks_total() -
                                  engine_->pool_blocks_in_use();
      DGPP_LOG_INFO(
          "sched: request '{}' deferred (needs {} blocks, {} free, {} "
          "slot(s) open) — admits when a peer retires",
          head->spec.id, reserve_blocks(*head), free_blocks,
          std::count(slots_.begin(), slots_.end(), -1));
    }
  }

  // (2) One physical decode pass per tick, over the next round-robin slice.
  // Scalar engines advertise capacity one and retain the original policy
  // bit-for-bit. A batched graph normally advertises the slot count, so this
  // slice contains every active request exactly once.
  std::vector<int> step_arrivals;
  step_arrivals.reserve(static_cast<size_t>(decode_batch_capacity_));
  for (int off = 1;
       off <= static_cast<int>(requests_.size()) &&
       static_cast<int>(step_arrivals.size()) < decode_batch_capacity_;
       ++off) {
    const int i = (cursor_ + off) % static_cast<int>(requests_.size());
    if (requests_[static_cast<size_t>(i)].state == State::kActive)
      step_arrivals.push_back(i);
  }
  if (!step_arrivals.empty()) {
    // The prefix cache's rolling snapshots (M7) sit right before the step:
    // a live request at an aligned committed position keeps its state in
    // its rolling slot (the close-time entry's source) — every rank at the
    // same quantum, since the position is journaled state.
    rolling_snapshots();
    step_batch(step_arrivals);
    progressed = true;
  }

  if (!progressed) {
    // No admission, no step, work remaining. With any active request
    // the rotation always yields one, so this is the admission
    // deadlock: the queue cannot fit the pool ever (its head's
    // reservation exceeds the total capacity) or every slot is held
    // by requests that can never retire (impossible under full-reserve
    // — they are bounded by max_steps). Either way: loud.
    const auto head = std::find_if(
        requests_.begin(), requests_.end(),
        [](const Request& r) { return r.state == State::kQueued; });
    throw std::runtime_error(
        "Scheduler: admission deadlock — request '" + head->spec.id +
        "' needs " + std::to_string(reserve_blocks(*head)) +
        " blocks against a pool of " +
        std::to_string(engine_->pool_blocks_total()) +
        " (grow --kv-capacity or shed requests)");
  }
  return true;
}

void Scheduler::run_to_completion() {
  while (tick()) {
  }
}

Scheduler::Meters Scheduler::meters() const {
  Meters m;
  for (const Request& r : requests_) {
    if (r.state == State::kActive) ++m.active;
    else if (r.state == State::kPrefilling) { ++m.active; ++m.prefilling; }
    else if (r.state == State::kQueued) ++m.queued;
    m.record_tokens +=
        static_cast<int64_t>(r.spec.prompt.size() + r.generated.size());
  }
  for (const Result& res : results_)
    m.record_tokens += static_cast<int64_t>(res.generated.size());
  m.terminal = static_cast<int>(retired_);
  m.records = static_cast<int64_t>(requests_.size());
  m.pool_blocks_total = engine_->pool_blocks_total();
  m.pool_blocks_in_use = engine_->pool_blocks_in_use();
  m.tokens_generated = tokens_generated_;
  m.reservations_grown = grows_;
  m.requests_shed_pool = pool_sheds_;
  m.prompts_prefilled = prompts_prefilled_;
  m.prompt_tokens = prompt_tokens_;
  m.prompt_tokens_computed = prompt_tokens_computed_;
  m.decode_steps = decode_steps_;
  m.decode_rows = decode_rows_;
  m.prefill_ms = prefill_ms_;
  m.prefill_request_ms = prefill_request_ms_;
  m.step_ms = step_ms_;
  m.mtp = engine_->mtp_acceptance();
  m.decode_batch = engine_->decode_batch_stats();
  m.prefix_slots = cache_.slots();
  m.prefix_entries = cache_.live_entries();
  m.prefix_hits = cache_.stats().hits;
  m.prefix_misses = cache_.stats().misses;
  m.prefix_tokens_saved = cache_.stats().tokens_saved;
  m.prefix_snapshots = cache_.stats().snapshots;
  m.prefix_head_snapshots = cache_.stats().head_snapshots;
  m.prefix_close_entries = cache_.stats().close_entries;
  m.prefix_rolling = cache_.stats().rolling;
  m.prefix_hops = cache_.stats().hops;
  m.prefix_evictions = cache_.stats().evictions;
  m.prefix_duplicates = cache_.stats().duplicates;
  m.prefix_skipped = cache_.stats().skipped_no_slot;
  m.prefix_skipped_no_block = cache_.stats().skipped_no_block;
  m.prefix_skipped_image_bytes = cache_.stats().skipped_image_bytes;
  m.prefix_image_bytes = static_cast<int64_t>(cache_.image_bytes());
  m.prefix_blocks_pinned = cache_.blocks_pinned(prefix_info_.block_tokens);
  return m;
}

void Scheduler::rolling_snapshots() {
  if (!cache_.enabled()) return;
  const int64_t align = std::max<int64_t>(1, prefix_info_.align);
  // Blocks the hops armed in this pass will take inside the step, after
  // the snapshots below have taken theirs: each arm reserves its own.
  int64_t armed_blocks = 0;
  for (size_t i = 0; i < requests_.size(); ++i) {  // arrival order
    Request& r = requests_[i];
    if (r.state != State::kActive || !cache_on(r)) continue;
    // Committed tokens: the prompt plus every generated token but the last
    // (pending — the next step writes it).
    const int64_t committed =
        static_cast<int64_t>(r.spec.prompt.size()) + r.steps_done - 1;
    if (committed <= 0) continue;
    if (committed % align != 0) {
      // The hop (M7 under the two-token step): the next aligned position is
      // committed + 1 and a step that commits two tokens passes it without
      // stopping — arm the engine to take the state after its first row.
      const int64_t hop = committed + 1;
      if (prefix_info_.step_tokens_max < 2 || hop % align != 0) continue;
      if (hop <= r.attach_position || hop == r.rolling_position) continue;
      if (hop < policy_.prefix_min_tokens) continue;  // below the entry floor
      if (r.rolling_slot < 0) {
        const int slot = acquire_arena_slot(r.spec.id);
        if (slot < 0) {
          ++cache_.stats().skipped_no_slot;
          continue;
        }
        r.rolling_slot = slot;
      }
      const bool needs_block = snapshot_needs_block(hop, r.rolling_position);
      if (needs_block && !ensure_free_blocks(1 + armed_blocks, r.spec.id)) {
        ++cache_.stats().skipped_no_block;
        continue;
      }
      if (needs_block) ++armed_blocks;
      engine_->prefix_arm_hop(r.slot, r.rolling_slot, hop);
      r.hop_armed = hop;
      continue;
    }
    if (committed == r.rolling_position) continue;
    // A snapshot at a position the request attached at or below repeats an
    // entry that exists; one at or below the prefill-cut entry likewise.
    if (committed <= r.attach_position) continue;
    if (committed < policy_.prefix_min_tokens) continue;  // below the entry floor
    if (r.rolling_slot < 0) {
      const int slot = acquire_arena_slot(r.spec.id);
      if (slot < 0) {
        ++cache_.stats().skipped_no_slot;
        continue;
      }
      r.rolling_slot = slot;
    }
    if (snapshot_needs_block(committed, r.rolling_position) &&
        !ensure_free_blocks(1 + armed_blocks, r.spec.id)) {
      ++cache_.stats().skipped_no_block;
      continue;
    }
    engine_->prefix_snapshot(r.slot, r.rolling_slot, committed);
    r.rolling_position = committed;
    ++cache_.stats().rolling;
    cache_.note(4, static_cast<uint64_t>(committed),
                static_cast<uint64_t>(r.rolling_slot));
    emit_prefix(r.spec.id, "rolling", committed, r.rolling_slot);
  }
}

int Scheduler::acquire_arena_slot(const std::string& id) {
  int slot = cache_.take_free_slot();
  if (slot >= 0) return slot;
  slot = cache_.evict_lru();
  if (slot < 0) return -1;
  // The victim's state leaves the engine; the slot is the caller's now.
  engine_->prefix_release(slot);
  emit_prefix(id, "evict", 0, slot);
  return slot;
}

void Scheduler::free_arena_slot(int slot) {
  engine_->prefix_release(slot);
  cache_.give_back_slot(slot);
}

// A miss, explained at INFO: the live log of an agent session
// showed a 64,803-token turn arriving a third of a second after its
// predecessor retired and prefilling cold for 200 s, with the
// predecessor's entries present and unattached in the arena — the prompt's
// first 62K tokens had changed on the client. The line names how many cuts
// the lookup probed, how many entries the cache held, and where the prompt
// parts from the entry it shares the most with: a divergence inside the
// system prompt reads differently from one at the previous answer.
void Scheduler::log_prefix_miss(const Request& r) const {
  const int entries = cache_.live_entries();
  if (entries == 0) {
    DGPP_LOG_INFO("sched: request '{}' prefix cache miss — {} cut(s), the cache is empty",
                  r.spec.id, r.cuts.size());
    return;
  }
  const PrefixCache::Ghost ghost = cache_.ghost_at(r.cuts, r.cut_hashes);
  if (ghost.position > 0) {
    DGPP_LOG_INFO(
        "sched: request '{}' prefix cache miss — {} cut(s) probed against {} "
        "entries; an entry at this prompt's cut {} was evicted ({} eviction(s) "
        "ago, last used at tick {}, now tick {}; the arena holds {} slots)",
        r.spec.id, r.cuts.size(), entries, ghost.position,
        cache_.stats().evictions - ghost.eviction + 1, ghost.last_use, ticks_,
        cache_.slots());
    return;
  }
  const PrefixCache::Nearest near = cache_.nearest(r.spec.prompt, r.cache_images);
  if (near.entry < 0) return;
  const PrefixCache::Entry& e = cache_.entry(near.entry);
  const int64_t n = static_cast<int64_t>(r.spec.prompt.size());
  const char* reading =
      near.common == 0 ? "nothing in common: a different prompt from its first token"
      : near.common == e.position && e.next_token >= 0 && e.position < n &&
              e.next_token != r.spec.prompt[static_cast<size_t>(e.position)]
          ? "the prefix matches, but the MTP lookahead token at the cut changed"
      : near.common >= e.position ? "the whole entry, which sits at no cut of this prompt"
      : near.common >= n          ? "the whole prompt, a prefix of that entry: no entry at this "
                                    "prompt's own cuts (evicted, or never taken)"
                                  : "the prompt differs from it from that token on";
  DGPP_LOG_INFO(
      "sched: request '{}' prefix cache miss — {} cut(s) probed against {} "
      "entries; the nearest entry (position {}) shares the first {} of the "
      "prompt's {} tokens ({})",
      r.spec.id, r.cuts.size(), entries, e.position, near.common, n, reading);
}

void Scheduler::emit_prefix(const std::string& id, const char* op,
                            int64_t position, int slot) {
  DGPP_LOG_DEBUG("sched: prefix cache {} '{}' position {} slot {}", op, id,
                 position, slot);
  if (observer_) observer_->on_prefix(id, op, position, slot);
}

}  // namespace dgpp::sched
