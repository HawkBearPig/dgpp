#include "serve/rank_metrics.hpp"

#include <string>

namespace dgpp::serve {

void write_rank_metrics(prom::Writer& w, const RankIdentity& id,
                        const dgpp::sched::Scheduler::Meters& m, double snapshot_age_s,
                        int64_t ticks, const std::atomic<uint64_t>* collectives) {
  const std::string rank = "rank=\"" + std::to_string(id.rank) + "\"";
  w.gauge("dgpp_rank_info", "This rank's build and world; the value is always 1.", int64_t{1},
          rank + ",world_size=\"" + std::to_string(id.world) + "\",version=\"" + prom::escape_label(id.version) +
              "\",git_sha=\"" + prom::escape_label(id.git_sha) + "\"");
  w.gauge("dgpp_rank_snapshot_age_seconds",
          "Since this rank last published its meters (a peer publishes after every tick it applies).",
          snapshot_age_s, rank);
  if (ticks >= 0) w.counter("dgpp_rank_ticks_total", "Journal records this rank applied.", ticks, rank);
  w.counter("dgpp_rank_decode_steps_total", "Decode passes this rank ran.", m.decode_steps, rank);
  w.counter("dgpp_rank_decode_step_seconds_total",
            "This rank's wall time inside decode steps, collective waits included.", m.step_ms / 1000.0, rank);
  w.counter("dgpp_rank_prefill_seconds_total",
            "This rank's wall time inside prefill calls, collective waits included.", m.prefill_ms / 1000.0, rank);
  w.counter("dgpp_rank_generation_tokens_total", "Tokens this rank's scheduler generated.", m.tokens_generated,
            rank);
  w.gauge("dgpp_rank_num_requests_running", "Requests holding a slot on this rank.", int64_t{m.active}, rank);
  w.gauge("dgpp_rank_kv_pool_blocks_total", "This rank's K/V pool blocks.", m.pool_blocks_total, rank);
  w.gauge("dgpp_rank_kv_pool_blocks_in_use", "This rank's K/V pool blocks in use.", m.pool_blocks_in_use, rank);
  if (collectives != nullptr)
    w.counter("dgpp_rank_collectives_total", "Multi-rank collectives this rank completed.",
              collectives->load(std::memory_order_relaxed), rank);
}

RankMetricsServer::RankMetricsServer(uint16_t port, const std::string& bind_host, RankIdentity id,
                                     const std::atomic<uint64_t>* collectives)
    : id_(std::move(id)), collectives_(collectives), http_(port, this, /*max_connections=*/8, bind_host) {
  loop_ = std::thread([this] { http_.serve(); });
}

RankMetricsServer::~RankMetricsServer() {
  http_.stop();
  if (loop_.joinable()) loop_.join();
}

void RankMetricsServer::publish(const dgpp::sched::Scheduler::Meters& m, int64_t ticks) {
  std::lock_guard<std::mutex> lock(mutex_);
  meters_ = m;
  ticks_ = ticks;
  published_ = std::chrono::steady_clock::now();
}

void RankMetricsServer::handle(const HttpRequest& req, HttpResponseWriter& w) {
  if (req.path == "/health") {
    w.respond(200, "application/json", "{\"status\":\"ok\",\"rank\":" + std::to_string(id_.rank) + "}");
    return;
  }
  if (req.path != "/metrics/prometheus" && req.path != "/metrics") {
    w.respond(404, "text/plain; charset=utf-8", "rank metrics: GET /metrics/prometheus\n");
    return;
  }
  if (req.method != "GET") {
    w.respond(405, "text/plain; charset=utf-8", "use GET for metrics\n");
    return;
  }
  dgpp::sched::Scheduler::Meters m;
  int64_t ticks = 0;
  double age_s = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    m = meters_;
    ticks = ticks_;
    age_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - published_).count();
  }
  prom::Writer out("");
  write_rank_metrics(out, id_, m, age_s, ticks, collectives_);
  w.respond(200, "text/plain; version=0.0.4; charset=utf-8", out.take());
}

}  // namespace dgpp::serve
