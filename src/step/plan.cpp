// The stream plan's totals, checks and summary (step/plan.hpp).
#include "step/plan.hpp"

#include <algorithm>
#include <cstdio>
#include <map>

namespace dgpp::step {

namespace {

constexpr uint64_t kGrantAlign = 256;  // loaders/weight_build.hpp kWeightAllocAlign

uint64_t align_grant(uint64_t b) { return (b + kGrantAlign - 1) / kGrantAlign * kGrantAlign; }

// The weight of one range's reads in the step's expectation.
double read_weight(const LayerPlan& lp, const Range& r) {
  if (r.dep != Dependency::Router) return static_cast<double>(r.reads);
  if (lp.groups <= 0) return 0.0;
  return static_cast<double>(r.reads) * lp.expected_groups / static_cast<double>(lp.groups);
}

double range_step_bytes(const LayerPlan& lp, const Range& r) {
  return static_cast<double>(r.bytes_per_read()) * read_weight(lp, r);
}

std::string human(double bytes) {
  char buf[64];
  if (bytes >= 1e9)
    std::snprintf(buf, sizeof buf, "%.3f GB", bytes / 1e9);
  else if (bytes >= 1e6)
    std::snprintf(buf, sizeof buf, "%.2f MB", bytes / 1e6);
  else if (bytes >= 1e3)
    std::snprintf(buf, sizeof buf, "%.1f KB", bytes / 1e3);
  else
    std::snprintf(buf, sizeof buf, "%.0f B", bytes);
  return buf;
}

}  // namespace

ByteTotals& ByteTotals::operator+=(const ByteTotals& o) {
  fixed += o.fixed;
  token += o.token;
  router += o.router;
  selection += o.selection;
  return *this;
}

ByteTotals layer_step_bytes(const LayerPlan& lp) {
  ByteTotals t;
  for (const Range& r : lp.ranges) {
    const double b = range_step_bytes(lp, r);
    switch (r.dep) {
      case Dependency::Fixed: t.fixed += b; break;
      case Dependency::Token: t.token += b; break;
      case Dependency::Router: t.router += b; break;
      case Dependency::Selection: t.selection += b; break;
    }
  }
  return t;
}

ByteTotals step_bytes(const StepPlan& plan) {
  ByteTotals t;
  for (const LayerPlan& lp : plan.layers) t += layer_step_bytes(lp);
  t += layer_step_bytes(plan.globals);
  return t;
}

double consumer_step_bytes(const LayerPlan& lp, Consumer consumer) {
  double b = 0;
  for (const Range& r : lp.ranges)
    if (r.consumer == consumer) b += range_step_bytes(lp, r);
  return b;
}

double consumer_step_bytes(const StepPlan& plan, Consumer consumer) {
  double b = consumer_step_bytes(plan.globals, consumer);
  for (const LayerPlan& lp : plan.layers) b += consumer_step_bytes(lp, consumer);
  return b;
}

std::vector<std::string> plan_check(const LayerPlan& lp) {
  std::vector<std::string> problems;
  const std::string where = lp.layer < 0 ? std::string("globals") : "layer " + std::to_string(lp.layer);
  auto bad = [&](const std::string& what) { problems.push_back(where + ": " + what); };
  std::vector<const Range*> resident;
  for (size_t i = 0; i < lp.ranges.size(); ++i) {
    const Range& r = lp.ranges[i];
    const std::string tag = "'" + r.name + "'";
    if (r.bytes == 0) bad(tag + " has no bytes");
    if (r.read_bytes > r.bytes) bad(tag + " reads more than its bytes");
    if (r.phase < 0 || r.phase >= static_cast<int>(lp.phases.size())) bad(tag + " names no phase");
    if (r.reads > 0 && r.consumer == Consumer::None) bad(tag + " is read without a consumer");
    if (r.dep == Dependency::Router && (r.group < 0 || r.group >= lp.groups))
      bad(tag + " is router-bound without a group under " + std::to_string(lp.groups));
    if (r.scale_of >= 0) {
      if (r.scale_of >= static_cast<int>(lp.ranges.size()))
        bad(tag + " scales a range the plan does not hold");
      else {
        const Range& p = lp.ranges[static_cast<size_t>(r.scale_of)];
        if (p.scale_of >= 0) bad(tag + " scales a scale grid");
        if (p.group != r.group || p.matrix != r.matrix || p.dep != r.dep)
          bad(tag + " scales a payload of another group, matrix or class");
      }
    }
    if (r.source == Source::Resident) resident.push_back(&r);
  }
  std::sort(resident.begin(), resident.end(),
            [](const Range* a, const Range* b) { return a->offset < b->offset; });
  uint64_t cursor = 0;
  for (const Range* r : resident) {
    if (r->offset != cursor) {
      bad("'" + r->name + "' sits at offset " + std::to_string(r->offset) + ", the grant order expects " +
          std::to_string(cursor));
      cursor = r->offset;
    }
    cursor = align_grant(r->offset + r->bytes);
  }
  if (cursor != lp.span_bytes)
    bad("the resident ranges end at " + std::to_string(cursor) + ", the span holds " +
        std::to_string(lp.span_bytes));
  if (lp.groups > 0 && (lp.expected_groups < 0 || lp.expected_groups > lp.groups))
    bad("expects " + std::to_string(lp.expected_groups) + " of " + std::to_string(lp.groups) + " groups");
  return problems;
}

std::vector<std::string> plan_check(const StepPlan& plan) {
  std::vector<std::string> problems;
  for (const LayerPlan& lp : plan.layers) {
    auto p = plan_check(lp);
    problems.insert(problems.end(), p.begin(), p.end());
  }
  auto g = plan_check(plan.globals);
  problems.insert(problems.end(), g.begin(), g.end());
  return problems;
}

const char* to_string(Format f) {
  switch (f) {
    case Format::Bf16: return "bf16";
    case Format::F32: return "f32";
    case Format::Fp8E4m3: return "fp8";
    case Format::ScaleF32: return "scale_f32";
    case Format::ScaleE8m0: return "scale_e8m0";
    case Format::Fp4E2m1: return "fp4";
    case Format::Exl3Trellis: return "exl3";
    case Format::KvFp4Block: return "kv_fp4_block";
    case Format::KvFp8Block: return "kv_fp8_block";
    case Format::IndexKey: return "index_key";
    case Format::Bf12: return "bf12";
  }
  return "?";
}

const char* to_string(Layout l) {
  switch (l) {
    case Layout::RowMajor: return "row_major";
    case Layout::Tile16: return "tile16";
  }
  return "?";
}

const char* to_string(Staging s) {
  switch (s) {
    case Staging::Register: return "register";
    case Staging::Shared: return "shared";
    case Staging::L2: return "l2";
    case Staging::Direct: return "direct";
  }
  return "?";
}

const char* to_string(Dependency d) {
  switch (d) {
    case Dependency::Fixed: return "fixed";
    case Dependency::Token: return "token";
    case Dependency::Router: return "router";
    case Dependency::Selection: return "selection";
  }
  return "?";
}

const char* to_string(Source s) {
  switch (s) {
    case Source::Resident: return "resident";
    case Source::Pool: return "pool";
    case Source::Host: return "host";
  }
  return "?";
}

const char* to_string(Consumer c) {
  switch (c) {
    case Consumer::Norm: return "norm";
    case Consumer::HyperConn: return "hyper_conn";
    case Consumer::Embedding: return "embedding";
    case Consumer::Projection: return "projection";
    case Consumer::Indexer: return "indexer";
    case Consumer::Compressor: return "compressor";
    case Consumer::Attention: return "attention";
    case Consumer::Router: return "router";
    case Consumer::ExpertGateUp: return "expert_gate_up";
    case Consumer::ExpertDown: return "expert_down";
    case Consumer::SharedGateUp: return "shared_gate_up";
    case Consumer::SharedDown: return "shared_down";
    case Consumer::Head: return "head";
    case Consumer::DraftHead: return "draft_head";
    case Consumer::TableGather: return "table_gather";
    case Consumer::None: return "none";
  }
  return "?";
}

std::string plan_summary(const StepPlan& plan) {
  std::string out;
  const ByteTotals t = step_bytes(plan);
  out += plan.family + " rank " + std::to_string(plan.rank) + " of " + std::to_string(plan.world) + ", layout " +
         std::to_string(plan.layout_version) + ": " + std::to_string(plan.layers.size()) + " layers, step reads " +
         human(t.total()) + " (fixed " + human(t.fixed) + ", token " + human(t.token) + ", router " +
         human(t.router) + ", selection " + human(t.selection) + ")\n";
  std::map<Consumer, double> by_consumer;
  std::map<Consumer, size_t> ranges_of;
  auto tally = [&](const LayerPlan& lp) {
    for (const Range& r : lp.ranges) {
      by_consumer[r.consumer] += range_step_bytes(lp, r);
      ++ranges_of[r.consumer];
    }
  };
  for (const LayerPlan& lp : plan.layers) tally(lp);
  tally(plan.globals);
  out += "by consumer:\n";
  for (const auto& [c, b] : by_consumer) {
    char line[160];
    std::snprintf(line, sizeof line, "  %-16s %12s  %7zu ranges\n", to_string(c), human(b).c_str(), ranges_of[c]);
    out += line;
  }
  out += "layers:\n";
  auto describe = [&](const LayerPlan& lp) {
    const ByteTotals lt = layer_step_bytes(lp);
    char line[256];
    std::snprintf(line, sizeof line, "  %4d  span %12s  %6zu ranges  %2zu phases  step %10s  groups %.1f of %d\n",
                  lp.layer, human(static_cast<double>(lp.span_bytes)).c_str(), lp.ranges.size(), lp.phases.size(),
                  human(lt.total()).c_str(), lp.expected_groups, lp.groups);
    out += line;
  };
  for (const LayerPlan& lp : plan.layers) describe(lp);
  describe(plan.globals);
  return out;
}

}  // namespace dgpp::step
