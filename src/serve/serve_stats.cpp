#include "serve/serve_stats.hpp"

#include <algorithm>
#include <format>

#include "common/log.hpp"

namespace dgpp::serve {

ThroughputLog::ThroughputLog(double interval_s, int rank, bool mtp)
    : interval_s_(interval_s), rank_(rank), mtp_(mtp) {}

std::string ThroughputLog::observe(const Meters& m, const ServiceCounts* svc,
                                   Clock::time_point now) {
  if (!enabled()) return {};
  if (!primed_) {
    primed_ = true;
    last_ = now;
    prev_ = m;
    if (svc) prev_svc_ = *svc;
    return {};
  }
  const double seconds =
      std::chrono::duration<double>(now - last_).count();
  if (seconds < interval_s_) return {};
  const bool worked = m.decode_steps != prev_.decode_steps ||
                      m.prompts_prefilled != prev_.prompts_prefilled ||
                      (svc && svc->requests_total != prev_svc_.requests_total);
  const bool busy = worked || m.active > 0 || m.queued > 0;
  std::string line;
  if (busy || was_busy_) {
    line = format(rank_, seconds, prev_, m, svc ? &prev_svc_ : nullptr, svc, mtp_);
    DGPP_LOG_INFO("{}", line);
  }
  was_busy_ = busy;
  last_ = now;
  prev_ = m;
  if (svc) prev_svc_ = *svc;
  return line;
}

std::string ThroughputLog::format(int rank, double seconds, const Meters& prev,
                                  const Meters& cur,
                                  const ServiceCounts* prev_svc,
                                  const ServiceCounts* cur_svc, bool mtp) {
  const double s = seconds > 0.0 ? seconds : 1e-9;
  const auto rate = [&](int64_t n) { return static_cast<double>(n) / s; };
  const auto share = [&](double ms) {
    return 100.0 * ms / (s * 1000.0);
  };
  const auto per = [](double num, int64_t den) {
    return den > 0 ? num / static_cast<double>(den) : 0.0;
  };

  const int64_t prompts = cur.prompts_prefilled - prev.prompts_prefilled;
  const int64_t prompt_tokens = cur.prompt_tokens - prev.prompt_tokens;
  const int64_t computed = cur.prompt_tokens_computed - prev.prompt_tokens_computed;
  const int64_t saved = prompt_tokens - computed;
  const double prefill_ms = cur.prefill_ms - prev.prefill_ms;
  const int64_t steps = cur.decode_steps - prev.decode_steps;
  const int64_t rows = cur.decode_rows - prev.decode_rows;
  const int64_t generated = cur.tokens_generated - prev.tokens_generated;
  const double step_ms = cur.step_ms - prev.step_ms;
  const int64_t hits = cur.prefix_hits - prev.prefix_hits;
  const int64_t misses = cur.prefix_misses - prev.prefix_misses;

  // Decode first — what an operator watches: the tokens per second the
  // interval delivered (idle time included), the pace while decoding (step
  // time over generated tokens), the pass's wall time, and MTP's yield
  // (tokens per request-step; one draft per step, so the share of drafts
  // accepted is that yield less one).
  const double tok_per_row = per(static_cast<double>(generated), rows);
  std::string out = std::format(
      "stats: rank {} | {:.1f} s | decode {:.1f} tok/s, {:.1f} ms/tok, "
      "{:.1f} ms/step ({} step{} / {} tok, {:.0f} % of wall)",
      rank, seconds, rate(generated), per(step_ms, generated),
      per(step_ms, steps), steps, steps == 1 ? "" : "s", generated,
      share(step_ms));
  if (mtp && rows > 0) {
    // The yield, then the measured acceptance of each draft position over
    // the interval (2026-09-06: the engine counts them; depth 1 has p1).
    out += std::format(" | mtp {:.2f} tok/step/req", tok_per_row);
    std::string accept;
    for (int p = 0; p < cur.mtp.depth && p < 8; ++p) {
      const uint64_t attempts = cur.mtp.attempts[p] - prev.mtp.attempts[p];
      const uint64_t accepts = cur.mtp.accepts[p] - prev.mtp.accepts[p];
      if (attempts == 0) continue;
      accept += std::format(" p{} {:.0f} %", p + 1,
                            100.0 * static_cast<double>(accepts) /
                                static_cast<double>(attempts));
    }
    if (!accept.empty()) out += ", accept" + accept;
  }
  out += std::format(" | prefill {} prompt{} / {} tok", prompts,
                     prompts == 1 ? "" : "s", computed);
  if (prompts > 0)
    out += std::format(", {:.0f} tok/s, {:.2f} ms/tok, {:.0f} ms avg ({:.0f} % of wall)",
                       rate(computed), per(prefill_ms, computed),
                       per(prefill_ms, prompts), share(prefill_ms));
  else if (computed > 0)
    out += std::format(", {:.0f} tok/s, {:.2f} ms/tok ({:.0f} % of wall)",
                       rate(computed), per(prefill_ms, computed), share(prefill_ms));
  if (cur.prefix_slots > 0)
    out += std::format(", {} tok cached ({}/{} hit{})", saved, hits,
                       hits + misses, hits + misses == 1 ? "" : "s");
  out += std::format(" | live {}, queued {} | pool {}/{} blocks ({:.0f} %)",
                     cur.active, cur.queued, cur.pool_blocks_in_use,
                     cur.pool_blocks_total,
                     100.0 * per(static_cast<double>(cur.pool_blocks_in_use),
                                 cur.pool_blocks_total));
  if (cur.prefix_slots > 0)
    out += std::format(" | prefix cache {}/{} entries", cur.prefix_entries,
                       cur.prefix_slots);
  if (cur.disk_enabled)
    out += std::format(" | nvme cache {} entries, {:.1f}/{:.1f} GiB, {} restored, {} pending",
                       cur.disk_entries,
                       static_cast<double>(cur.disk_pages_used * cur.disk_page_bytes) /
                           (1024.0 * 1024.0 * 1024.0),
                       static_cast<double>(cur.disk_pages_total * cur.disk_page_bytes) /
                           (1024.0 * 1024.0 * 1024.0),
                       cur.disk_restored, cur.disk_pending);
  if (cur.prefilling > 0)
    out += std::format(" | {} prefilling", cur.prefilling);
  if (cur_svc && prev_svc) {
    out += std::format(
        " | requests +{} (shed {}, cancelled {})",
        cur_svc->requests_total - prev_svc->requests_total,
        cur_svc->requests_shed - prev_svc->requests_shed,
        cur_svc->requests_cancelled - prev_svc->requests_cancelled);
  }
  return out;
}

}  // namespace dgpp::serve
