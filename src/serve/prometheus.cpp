#include "serve/prometheus.hpp"

#include <charconv>
#include <cmath>
#include <stdexcept>

namespace dgpp::serve::prom {

Histogram::Histogram(std::vector<double> bounds) : bounds_(std::move(bounds)) {
  for (size_t i = 0; i < bounds_.size(); ++i) {
    if (!std::isfinite(bounds_[i]))
      throw std::invalid_argument("prometheus histogram: bucket bounds must be finite");
    if (i > 0 && !(bounds_[i] > bounds_[i - 1]))
      throw std::invalid_argument("prometheus histogram: bucket bounds must be strictly ascending");
  }
  buckets_.assign(bounds_.size() + 1, 0);
}

void Histogram::observe(double v) {
  if (std::isnan(v)) return;
  size_t i = 0;
  while (i < bounds_.size() && v > bounds_[i]) ++i;
  ++buckets_[i];
  ++count_;
  sum_ += v;
}

std::vector<double> latency_buckets() {
  return {0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0,
          7.5,   10.0,  20.0, 40.0,  80.0, 160.0, 320.0, 640.0};
}

std::vector<double> step_buckets() {
  return {0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05,
          0.075, 0.1,    0.15,  0.2,    0.3,  0.5,   1.0,  2.5,  5.0,  10.0};
}

std::vector<double> token_buckets() {
  return {1, 8, 32, 128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576};
}

std::string escape_label(std::string_view v) {
  std::string out;
  out.reserve(v.size());
  for (const char c : v) {
    if (c == '\\') out.append("\\\\");
    else if (c == '"') out.append("\\\"");
    else if (c == '\n') out.append("\\n");
    else out.push_back(c);
  }
  return out;
}

std::string format_value(double v) {
  if (std::isnan(v)) return "NaN";
  if (std::isinf(v)) return v > 0 ? "+Inf" : "-Inf";
  char buf[32];
  const auto [end, ec] = std::to_chars(buf, buf + sizeof(buf), v);
  (void)ec;
  return std::string(buf, end - buf);
}

namespace {

template <typename T>
std::string format_int(T v) {
  char buf[24];
  const auto [end, ec] = std::to_chars(buf, buf + sizeof(buf), v);
  (void)ec;
  return std::string(buf, end - buf);
}

}  // namespace

Writer::Writer(std::string base_labels) : base_(std::move(base_labels)) {}

void Writer::header(std::string_view name, const char* type, std::string_view help) {
  if (!declared_.emplace(name).second) return;
  out_.append("# HELP ").append(name).push_back(' ');
  out_.append(help).push_back('\n');
  out_.append("# TYPE ").append(name).push_back(' ');
  out_.append(type).push_back('\n');
}

void Writer::sample(std::string_view name, std::string_view suffix, std::string_view labels,
                    std::string_view extra, std::string_view value) {
  out_.append(name).append(suffix);
  if (!base_.empty() || !labels.empty() || !extra.empty()) {
    out_.push_back('{');
    bool first = true;
    for (const std::string_view part : {std::string_view(base_), labels, extra}) {
      if (part.empty()) continue;
      if (!first) out_.push_back(',');
      out_.append(part);
      first = false;
    }
    out_.push_back('}');
  }
  out_.push_back(' ');
  out_.append(value).push_back('\n');
}

void Writer::gauge(std::string_view name, std::string_view help, double v, std::string_view labels) {
  header(name, "gauge", help);
  sample(name, "", labels, {}, format_value(v));
}

void Writer::gauge(std::string_view name, std::string_view help, int64_t v, std::string_view labels) {
  header(name, "gauge", help);
  sample(name, "", labels, {}, format_int(v));
}

void Writer::counter(std::string_view name, std::string_view help, double v, std::string_view labels) {
  header(name, "counter", help);
  sample(name, "", labels, {}, format_value(v));
}

void Writer::counter(std::string_view name, std::string_view help, uint64_t v, std::string_view labels) {
  header(name, "counter", help);
  sample(name, "", labels, {}, format_int(v));
}

void Writer::counter(std::string_view name, std::string_view help, int64_t v, std::string_view labels) {
  header(name, "counter", help);
  sample(name, "", labels, {}, format_int(v < 0 ? 0 : v));
}

void Writer::histogram(std::string_view name, std::string_view help, const Histogram& h,
                       std::string_view labels) {
  header(name, "histogram", help);
  uint64_t cumulative = 0;
  const auto& bounds = h.bounds();
  const auto& buckets = h.buckets();
  for (size_t i = 0; i <= bounds.size(); ++i) {
    cumulative += buckets[i];
    const std::string le =
        "le=\"" + (i < bounds.size() ? format_value(bounds[i]) : std::string("+Inf")) + "\"";
    sample(name, "_bucket", labels, le, format_int(cumulative));
  }
  sample(name, "_sum", labels, {}, format_value(h.sum()));
  sample(name, "_count", labels, {}, format_int(h.count()));
}

}  // namespace dgpp::serve::prom
