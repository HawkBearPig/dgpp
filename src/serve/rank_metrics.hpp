#pragma once
// Per-rank metrics: what each rank of a world measures of its own work.
//
// Rank 0's /metrics/prometheus describes the service — requests, tokens,
// latencies — once for the whole world. The ranks execute every step
// together, but each spends its own time in it: its decode steps and
// prefills include the collectives it waited on, so a rank whose step time
// runs ahead of the others is the slow one (a throttled GPU, a congested
// link). The dgpp_rank_* families report that per rank, labeled `rank`.
// Rank 0 appends them to its exposition; a peer, which serves no HTTP API,
// serves them on its own metrics listener (`ports.metrics`, on its node
// address) when one is configured.
//
// The collective bus's per-lane traffic counters are updated outside its
// stats lock, so they are not read here; the completion epoch is atomic and
// is.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

#include "sched/scheduler.hpp"
#include "serve/http_server.hpp"
#include "serve/prometheus.hpp"

namespace dgpp::serve {

struct RankIdentity {
  int rank = 0;
  int world = 1;
  std::string version;
  std::string git_sha;
};

// Writes the dgpp_rank_* families for one rank. `snapshot_age_s`: since the
// meters were published; `ticks`: journal records applied (peers), or -1 to
// omit the family (rank 0 has no journal to follow); `collectives`: the
// bus's completion epoch, or null without a bus.
void write_rank_metrics(prom::Writer& w, const RankIdentity& id,
                        const dgpp::sched::Scheduler::Meters& m, double snapshot_age_s,
                        int64_t ticks, const std::atomic<uint64_t>* collectives);

// A peer's metrics listener: GET /metrics/prometheus (also /metrics) and
// GET /health on `bind_host:port`, served from the last published snapshot
// by a thread of its own, so a scrape never touches the engine loop.
class RankMetricsServer final : public HttpHandler {
 public:
  // Binds at once (throws std::runtime_error naming the address when it
  // cannot) and starts serving.
  RankMetricsServer(uint16_t port, const std::string& bind_host, RankIdentity id,
                    const std::atomic<uint64_t>* collectives);
  ~RankMetricsServer() override;
  RankMetricsServer(const RankMetricsServer&) = delete;
  RankMetricsServer& operator=(const RankMetricsServer&) = delete;

  uint16_t port() const { return http_.port(); }

  // The peer loop, after every tick.
  void publish(const dgpp::sched::Scheduler::Meters& m, int64_t ticks);

  void handle(const HttpRequest& req, HttpResponseWriter& w) override;

 private:
  RankIdentity id_;
  const std::atomic<uint64_t>* collectives_;
  std::mutex mutex_;
  dgpp::sched::Scheduler::Meters meters_;
  int64_t ticks_ = 0;
  std::chrono::steady_clock::time_point published_ = std::chrono::steady_clock::now();
  HttpServer http_;
  std::thread loop_;
};

}  // namespace dgpp::serve
