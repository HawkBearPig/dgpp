#pragma once
// Prometheus text exposition (format 0.0.4) for GET /metrics/prometheus.
//
// Two pieces, both host-only and allocation-light: a fixed-bucket
// histogram the service observes request latencies and sizes into (under
// its own lock — the histogram itself is not synchronized), and a writer
// that renders metric families with one HELP/TYPE header per name. Every
// sample carries the writer's base labels (the served model) ahead of its
// own; label values are escaped as the format requires.
//
// The JSON metrics (/metrics, /v1/metrics) stay the complete operator
// view; the exposition renders the same counters with Prometheus types,
// plus the distributions JSON cannot carry.
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace dgpp::serve::prom {

class Histogram {
 public:
  // `bounds`: the buckets' inclusive upper bounds, strictly ascending and
  // finite; the +Inf bucket is implicit. Throws std::invalid_argument
  // otherwise.
  explicit Histogram(std::vector<double> bounds);

  // Counts `v` into the first bucket whose bound is >= v (NaN is ignored).
  void observe(double v);

  const std::vector<double>& bounds() const { return bounds_; }
  // Per-bucket counts, NOT cumulative: bounds().size() + 1 entries, the
  // last one the +Inf overflow. The writer accumulates them.
  const std::vector<uint64_t>& buckets() const { return buckets_; }
  uint64_t count() const { return count_; }
  double sum() const { return sum_; }

 private:
  std::vector<double> bounds_;
  std::vector<uint64_t> buckets_;
  uint64_t count_ = 0;
  double sum_ = 0.0;
};

// Seconds, 1 ms to 10 min: time to first token on a 256K-token prompt runs
// to minutes, an attached prefix answers in milliseconds.
std::vector<double> latency_buckets();
// Seconds, 1 ms to 10 s: a decode step or the gap between token batches.
std::vector<double> step_buckets();
// Tokens per request, 1 to 1M.
std::vector<double> token_buckets();

// Escapes a label value: backslash, double quote and newline.
std::string escape_label(std::string_view v);

class Writer {
 public:
  // `base_labels`: rendered `name="value"` pairs (comma-separated, no
  // braces) every sample carries first; may be empty.
  explicit Writer(std::string base_labels);

  // `labels`: the sample's own `name="value"` pairs, appended after the
  // base labels. Samples of one family must be written consecutively.
  void gauge(std::string_view name, std::string_view help, double v, std::string_view labels = {});
  void counter(std::string_view name, std::string_view help, double v, std::string_view labels = {});
  // Integer forms: exact past 2^53 (cumulative token counts get there).
  void gauge(std::string_view name, std::string_view help, int64_t v, std::string_view labels = {});
  void counter(std::string_view name, std::string_view help, uint64_t v, std::string_view labels = {});
  void counter(std::string_view name, std::string_view help, int64_t v, std::string_view labels = {});
  void histogram(std::string_view name, std::string_view help, const Histogram& h,
                 std::string_view labels = {});
  // Appends a sample verbatim, unlabeled and without the base labels (the
  // pre-existing unprefixed spec_decode lines keep their exact shape).
  void raw(std::string_view text) { out_.append(text); }

  std::string take() { return std::move(out_); }

 private:
  void header(std::string_view name, const char* type, std::string_view help);
  void sample(std::string_view name, std::string_view suffix, std::string_view labels,
              std::string_view extra, std::string_view value);

  std::string base_;
  std::string out_;
  std::unordered_set<std::string> declared_;
};

// Shortest round-trip form; +Inf / -Inf / NaN as the format spells them.
std::string format_value(double v);

}  // namespace dgpp::serve::prom
