#include "serve/prometheus.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"

using dgpp::serve::prom::escape_label;
using dgpp::serve::prom::format_value;
using dgpp::serve::prom::Histogram;
using dgpp::serve::prom::Writer;

namespace {

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error(what);
}

size_t count_of(const std::string& hay, const std::string& needle) {
  size_t n = 0;
  for (size_t at = hay.find(needle); at != std::string::npos; at = hay.find(needle, at + 1)) ++n;
  return n;
}

}  // namespace

DGPP_TEST(prometheus_histogram_bucketsAreUpperInclusive) {
  Histogram h({0.1, 1.0});
  for (const double v : {0.05, 0.1, 0.5, 1.0, 3.0}) h.observe(v);
  h.observe(std::nan(""));  // ignored
  require(h.buckets().size() == 3, "two bounds plus +Inf");
  require(h.buckets()[0] == 2 && h.buckets()[1] == 2 && h.buckets()[2] == 1, "le is inclusive");
  require(h.count() == 5 && std::abs(h.sum() - 4.65) < 1e-12, "count and sum skip NaN");
}

DGPP_TEST(prometheus_histogram_rejectsBadBounds) {
  for (const auto& bounds : {std::vector<double>{1.0, 1.0}, std::vector<double>{2.0, 1.0},
                             std::vector<double>{1.0, std::numeric_limits<double>::infinity()}}) {
    bool threw = false;
    try {
      Histogram h(bounds);
    } catch (const std::invalid_argument& e) {
      threw = std::string(e.what()).find("bucket bounds") != std::string::npos;
    }
    require(threw, "invalid bounds are refused, naming the bounds");
  }
}

DGPP_TEST(prometheus_writer_rendersFamiliesOnce) {
  Writer w("model_name=\"m\"");
  w.counter("x_total", "An x.", uint64_t{18446744073709551615ull}, "kind=\"a\"");
  w.counter("x_total", "An x.", int64_t{-3}, "kind=\"b\"");
  w.gauge("g", "A g.", 0.25);
  Histogram h({1.0, 2.0});
  h.observe(1.5);
  w.histogram("lat_seconds", "A latency.", h, "path=\"p\"");
  const std::string out = w.take();
  require(count_of(out, "# TYPE x_total counter\n") == 1 && count_of(out, "# HELP x_total An x.\n") == 1,
          "one header per family: " + out);
  require(out.find("x_total{model_name=\"m\",kind=\"a\"} 18446744073709551615\n") != std::string::npos,
          "integer counters are exact: " + out);
  require(out.find("x_total{model_name=\"m\",kind=\"b\"} 0\n") != std::string::npos, "counters never go negative");
  require(out.find("g{model_name=\"m\"} 0.25\n") != std::string::npos, "gauge with base labels only");
  require(out.find("lat_seconds_bucket{model_name=\"m\",path=\"p\",le=\"1\"} 0\n") != std::string::npos &&
              out.find("lat_seconds_bucket{model_name=\"m\",path=\"p\",le=\"2\"} 1\n") != std::string::npos &&
              out.find("lat_seconds_bucket{model_name=\"m\",path=\"p\",le=\"+Inf\"} 1\n") != std::string::npos,
          "cumulative buckets, le last: " + out);
  require(out.find("lat_seconds_sum{model_name=\"m\",path=\"p\"} 1.5\n") != std::string::npos &&
              out.find("lat_seconds_count{model_name=\"m\",path=\"p\"} 1\n") != std::string::npos,
          "sum and count: " + out);
}

DGPP_TEST(prometheus_writer_unlabeledAndEscaped) {
  Writer w("");
  w.gauge("u", "Unlabeled.", int64_t{7});
  require(w.take() == "# HELP u Unlabeled.\n# TYPE u gauge\nu 7\n", "no braces without labels");
  require(escape_label("a\"b\\c\nd") == "a\\\"b\\\\c\\nd", "label escaping");
  require(format_value(std::numeric_limits<double>::infinity()) == "+Inf" && format_value(std::nan("")) == "NaN",
          "special values");
}
