// DeepSeek-V4.1's arena placement (step/dsv41/arena.hpp).
#include "step/dsv41/arena.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "models/dsv41/loader.hpp"

namespace dgpp::step {

namespace {

constexpr uint64_t kGrantAlign = 256;  // loaders/weight_build.hpp kWeightAllocAlign
uint64_t align_grant(uint64_t b) { return (b + kGrantAlign - 1) / kGrantAlign * kGrantAlign; }

int class_rank(Dependency d) {
  switch (d) {
    case Dependency::Fixed: return 0;
    case Dependency::Token: return 1;
    case Dependency::Selection: return 2;
    case Dependency::Router: return 3;
  }
  return 4;
}

}  // namespace

std::vector<size_t> dsv41_arena_order(const LayerPlan& lp) {
  std::vector<size_t> order;
  for (size_t i = 0; i < lp.ranges.size(); ++i)
    if (lp.ranges[i].source == Source::Resident) order.push_back(i);
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    const Range& x = lp.ranges[a];
    const Range& y = lp.ranges[b];
    if (x.phase != y.phase) return x.phase < y.phase;
    if (class_rank(x.dep) != class_rank(y.dep)) return class_rank(x.dep) < class_rank(y.dep);
    if (x.dep == Dependency::Router) {
      if (x.group != y.group) return x.group < y.group;
      if (x.matrix != y.matrix) return x.matrix < y.matrix;
      if ((x.scale_of >= 0) != (y.scale_of >= 0)) return x.scale_of < y.scale_of;  // payload first
    }
    return x.offset < y.offset;  // grant order
  });
  return order;
}

std::vector<size_t> dsv41_arena_placement(const Dsv41TextConfig& cfg, int layer, int rank, int world) {
  // The grant-order layout, then its ranges: each resident range is one
  // grant, and the plan's tiling check guarantees the two agree.
  void* base = reinterpret_cast<void*>(static_cast<uintptr_t>(1) << 40);
  Dsv41LayerResident r;
  const size_t span = Dsv41LayerStream::layout_layer(cfg, layer, rank, world, base, r);
  const Dsv41LocalGeometry geo = Dsv41LocalGeometry::from_config(cfg, rank, world, LoaderHeadSharding::Full);
  Dsv41StepShape shape;
  shape.rows = 1;
  shape.draft_rows = 1;
  const LayerPlan lp = dsv41_layer_plan(cfg, geo, r, base, span, shape);
  {
    const std::vector<std::string> problems = plan_check(lp);
    if (!problems.empty())
      throw std::logic_error("dsv41 arena: the plan of layer " + std::to_string(layer) + " is not sound: " + problems.front());
  }
  // Grant ordinals: the resident ranges in offset order.
  std::vector<size_t> by_offset;
  for (size_t i = 0; i < lp.ranges.size(); ++i)
    if (lp.ranges[i].source == Source::Resident) by_offset.push_back(i);
  std::sort(by_offset.begin(), by_offset.end(),
            [&](size_t a, size_t b) { return lp.ranges[a].offset < lp.ranges[b].offset; });
  std::vector<size_t> ordinal_of(lp.ranges.size(), SIZE_MAX);
  for (size_t k = 0; k < by_offset.size(); ++k) ordinal_of[by_offset[k]] = k;
  // The arena: the ranges in consumption order, each at the next grant.
  std::vector<size_t> placement(by_offset.size(), SIZE_MAX);
  uint64_t cursor = 0;
  for (const size_t i : dsv41_arena_order(lp)) {
    placement[ordinal_of[i]] = cursor;
    cursor = align_grant(cursor + lp.ranges[i].bytes);
  }
  if (cursor != span)
    throw std::logic_error("dsv41 arena: layer " + std::to_string(layer) + " places " + std::to_string(cursor) +
                           " bytes in a span of " + std::to_string(span));
  return placement;
}

ResidentLayout dsv41_resident_layout(const Dsv41TextConfig& cfg, int rank, int world) {
  ResidentLayout layout;
  layout.version = kDsv41ArenaVersion;
  layout.placement = [cfg, rank, world](int layer) { return dsv41_arena_placement(cfg, layer, rank, world); };
  return layout;
}

}  // namespace dgpp::step
