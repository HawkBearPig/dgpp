#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace dgpp::sched {

inline constexpr int kDefaultDecodeGapTargetMs = 1000;
inline constexpr int kDefaultFirstTokenTargetMs = 3000;
inline constexpr int kMaxLatencyTargetMs = 3600000;
inline constexpr int64_t kInitialPrefillBudgetTokens = 256;
inline constexpr int kDefaultPrefillIdleFloorTokens = 1024;

struct PrefillLatencyTuning {
  int headroom_pct = 15;  // reserve this fraction of the time target
  int growth_factor = 2;  // maximum upward probe relative to observed work
  int recovery_pct = 25;  // weight of a cheaper observation in the cost estimate
  int initial_tokens = kInitialPrefillBudgetTokens;
  // The smallest quantum selected while nothing decodes and nothing waits
  // for a free slot (2026-10-15). A lone reader has no decode gap to
  // protect — the only party the first-token target serves is an arrival
  // that may come during its chunk — and below this size the engine's
  // prefill forms get expensive: the dense MLP leaves its prefill GEMM, the
  // weights stream once per chunk, the attention grid no longer fills the
  // SMs. A 522K prompt on the 27B ran its tail in 256-token chunks and took
  // 39 % longer than at 4096. The floor costs that arrival at most one
  // floor-sized chunk of wait at extreme context (first_token_target_misses
  // counts it). Applied once a chunk cost is measured; 0: the alignment.
  int idle_floor_tokens = kDefaultPrefillIdleFloorTokens;
  bool valid() const {
    return headroom_pct >= 0 && headroom_pct <= 50 && growth_factor >= 2 && growth_factor <= 4 &&
           recovery_pct >= 1 && recovery_pct <= 100 && initial_tokens >= 1 && initial_tokens <= (1 << 30) &&
           idle_floor_tokens >= 0 && idle_floor_tokens <= (1 << 30);
  }
  bool operator==(const PrefillLatencyTuning&) const = default;
};

// A deployment learns its own cost; no checkpoint or quantization lookup.
// Only the leader consults this feedback. Peers replay the selected token
// limit, never their own clocks. The caller supplies completed pass durations
// including decode, snapshots and other work that delays the next yield.
//
// The decoders' own pass (2026-10-22): a mixed pass carries the decode rows
// whether or not it reads anything, so its duration is the decoders' step
// plus what the chunk adds. Charging the whole pass to the chunk's tokens
// made a small chunk look dear per token, and the limit it produced made
// the next one dearer still: growth is bounded at twice the last quantum,
// so a quantum grows only when its pass is under half the allowance, and
// with decoders live every quantum whose pass ran between 42.5 % and 85 %
// of the target was an absorbing state — one slow pass walked the limit
// down the ladder for good. A field report (one Spark, five opencode
// agents, 2026-10-21) read the limit at 128 tokens — 32 per reader — for
// whole runs, at twice the idle prompt cost per token. The controller now
// learns the decoders' pass from decode-only ticks at each load and charges
// the chunk only for what it adds; the target less that pass is the room
// for tokens. A lone reader has no decoders' pass and keeps its idle floor.
//
// The cost is kept per decoder-load range AND per context range
// (2026-10-15): a chunk's cost per token grows with the position it starts
// at (the full-attention families read every earlier key), so one estimate
// for all positions let a 512K prompt's tail — 7 ms a token — size the next
// 4K prompt's chunks at 256 tokens for the rest of its prefill (2.8x its
// first-token time). Each range learns its own rate; an unmeasured range
// borrows the most conservative rate measured at the same context under
// another load, then the deepest shallower context's.
class PrefillLatencyController {
 public:
  // `context`: the position the measured chunk started at (the deepest
  // reader's), which picks the range the observation trains.
  void observe(int decoders, int64_t context, int64_t tokens, int64_t offered, double ms,
               const PrefillLatencyTuning& tuning = {}) {
    // Final tails and snapshot cuts are not representative full quanta.
    if (tokens <= 0 || tokens < offered / 2 || !(ms > 0) || !std::isfinite(ms)) return;
    auto& sample = samples_[bucket(decoders)][context_bucket(context)];
    // What the chunk added to the pass the decoders would have run anyway.
    const double rate = std::max(0.0, ms - decode_pass(decoders)) / static_cast<double>(tokens);
    // React immediately to a slowdown, recover gradually as work gets cheaper.
    const double weight = tuning.recovery_pct / 100.0;
    sample.ms_per_token = sample.tokens == 0 || rate > sample.ms_per_token
        ? rate : (1 - weight) * sample.ms_per_token + weight * rate;
    sample.tokens = tokens;
  }

  // A tick that only stepped `decoders` (nothing read, nothing admitted):
  // the pass a mixed tick at this load pays before it reads a token.
  // Cheaper at once, dearer gradually: an overestimate oversizes the next
  // chunk and misses the target; an underestimate only charges the chunk
  // for a little of the decoders' work, which is where the estimate stood
  // before this pass was measured at all.
  void observe_decode(int decoders, double ms, const PrefillLatencyTuning& tuning = {}) {
    if (decoders <= 0 || !(ms > 0) || !std::isfinite(ms)) return;
    double& pass = decode_ms_[bucket(decoders)];
    const double weight = tuning.recovery_pct / 100.0;
    pass = pass == 0 || ms < pass ? ms : (1 - weight) * pass + weight * ms;
  }

  // The learned decoders' pass at this load (0: none measured, or no decoders).
  double decode_pass(int decoders) const {
    if (decoders <= 0) return 0;
    // Own load first, then the nearest measured load below (a lighter
    // batch steps for less), then the nearest above.
    const size_t own = bucket(decoders);
    for (size_t b = own + 1; b-- > 1;)
      if (decode_ms_[b] > 0) return decode_ms_[b];
    for (size_t b = own + 1; b < kLoads; ++b)
      if (decode_ms_[b] > 0) return decode_ms_[b];
    return 0;
  }

  // `context`: the position the next chunk starts at; `floor`: the smallest
  // quantum to select (0: the alignment), aligned down and capped.
  int64_t limit(int decoders, int64_t context, double target_ms, int64_t initial, int64_t capacity,
                int64_t alignment, int64_t floor, int64_t wave = 0,
                const PrefillLatencyTuning& tuning = {}) const {
    const int64_t cap = capacity / alignment * alignment;
    if (!(target_ms > 0)) return cap;
    const int64_t low = floor > 0 ? std::min(cap, std::max(alignment, floor / alignment * alignment)) : alignment;
    const Sample* sample = pick(bucket(decoders), context_bucket(context));
    // The cold start keeps the cautious initial quantum (the busy budget):
    // the floor applies once a chunk's cost has been measured.
    if (!sample) return waves(std::min(cap, std::max(alignment, initial / alignment * alignment)), cap, alignment, wave);
    // Leave room for cost variation and arrivals. Bound upward exploration;
    // the first unmeasured workload starts at the shared busy budget. The
    // decoders' pass comes out of the allowance before any token does.
    const double room = (1 - tuning.headroom_pct / 100.0) * target_ms - decode_pass(decoders);
    double predicted = 0;
    if (room > 0) predicted = sample->ms_per_token > 0 ? std::min<double>(cap, room / sample->ms_per_token) : cap;
    const int64_t rows = std::min<int64_t>(static_cast<int64_t>(predicted), sample->tokens * tuning.growth_factor);
    const int64_t bound = std::min(cap, std::max(alignment, rows / alignment * alignment));
    // A small geometric set avoids oscillating among new GEMM shapes (and
    // repeatedly paying their first-use setup). Preserve an aligned capacity
    // that is not a power of two when the estimate can afford it.
    int64_t quantum = alignment;
    while (quantum <= bound / 2) quantum *= 2;
    return waves(std::min(cap, std::max(bound == cap ? cap : quantum, low)), cap, alignment, wave);
  }

 private:
  // The engine's attention grid runs in waves of `wave` rows: a quantum
  // below the cap rounds up to whole waves (2026-10-15), never past the cap.
  static int64_t waves(int64_t chosen, int64_t cap, int64_t alignment, int64_t wave) {
    if (wave <= 0 || chosen >= cap) return chosen;
    const int64_t w = wave / alignment * alignment;
    if (w <= 0) return chosen;
    return std::min(cap, (chosen + w - 1) / w * w);
  }
  static constexpr size_t kLoads = 6, kContexts = 5;
  static size_t bucket(int decoders) {
    return decoders <= 2 ? static_cast<size_t>(std::max(0, decoders))
                        : decoders <= 4 ? 3 : decoders <= 8 ? 4 : 5;
  }
  // The chunk's start position in steps of four from 8K: below 8K, 32K,
  // 128K, 512K, and beyond.
  static size_t context_bucket(int64_t context) {
    size_t b = 0;
    for (int64_t edge = 8192; b + 1 < kContexts && context >= edge; edge *= 4) ++b;
    return b;
  }
  struct Sample { double ms_per_token = 0; int64_t tokens = 0; };
  // The estimate for (load, context): its own; else the most conservative
  // at the same context under another load (a load changes a chunk's cost
  // by the decode rows it shares, a context by every key it reads); else
  // the deepest shallower context, own load first, any load next — a
  // shallower rate under-predicts, and the growth bound and the immediate
  // slowdown rule correct it within a chunk.
  const Sample* pick(size_t load, size_t ctx) const {
    for (size_t c = ctx + 1; c-- > 0;) {
      if (samples_[load][c].tokens) return &samples_[load][c];
      const Sample* worst = nullptr;
      for (const auto& row : samples_)
        if (row[c].tokens && (!worst || row[c].ms_per_token > worst->ms_per_token)) worst = &row[c];
      if (worst) return worst;
    }
    return nullptr;
  }
  std::array<std::array<Sample, kContexts>, kLoads> samples_{};
  std::array<double, kLoads> decode_ms_{};  // the decoders' own pass per load; 0 unmeasured
};

}  // namespace dgpp::sched
