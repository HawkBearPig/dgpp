// The stream plan's lowering (step/device_plan.hpp).
#include "step/device_plan.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace dgpp::step {

namespace {

constexpr uint64_t kGrantAlign = 256;
constexpr int kStepRows = 32;  // kernels/step/stream.hpp kStepMaxRows: the projection buffer's rows
uint64_t align_grant(uint64_t b) { return (b + kGrantAlign - 1) / kGrantAlign * kGrantAlign; }

// A projection's units: 128 output rows each (eight warps on 16-row tiles;
// 64-row units measured slower, each unit's staging and its halved loads in
// flight costing more than the finer tail saved) over the inner width in as
// many slabs as leave a slab of at least 1280 codes dividing the windows
// evenly, so a unit is about 160 KB and the phase's tail a fraction of its
// time. Slabbed pools reduce their fp32 partials in slab order afterwards.
constexpr int32_t kProjectionUnitRows = 128;
// The slabs of a projection's inner width: the most that leave a slab of at
// least 640 codes (two and a half windows). Units of 80 to 130 KB give a
// phase of a few matrices four or more waves over the workers, so its tail
// is a fraction of a unit rather than most of one (2026-10-30: at 1280-code
// slabs the attn_out phase drained for half its length).
int32_t projection_slabs(int k) {
  int32_t best = 1;
  for (int d = 2; d <= 8; ++d)
    if ((k / 256) % d == 0 && k / d >= 640) best = d;
  return best;
}

int add_pool(DevicePlan& dp, DevPhase& d, const DevPool& pool) {
  if (d.n_pools >= kMaxPhasePools) throw std::logic_error("lower_plan: a phase has more pools than the kernel holds");
  const int index = static_cast<int>(dp.pools.size());
  dp.pools.push_back(pool);
  d.pools[d.n_pools] = index;
  if (pool.group_kind == kGroupRouted) d.routed = static_cast<int8_t>(d.n_pools);
  ++d.n_pools;
  dp.max_k = std::max(dp.max_k, pool.k);
  return index;
}

}  // namespace

DevicePlan lower_plan(const std::vector<LayerPlan>& layers, uint32_t unit_bytes, bool consumers) {
  if (unit_bytes == 0 || unit_bytes % 512 != 0) throw std::invalid_argument("lower_plan: unit_bytes must be a multiple of 512");
  DevicePlan dp;
  dp.unit_bytes = unit_bytes;
  uint32_t lead = 0;
  for (const LayerPlan& lp : layers) {
    DevLayer dl;
    dl.phase_begin = static_cast<int32_t>(dp.phases.size());
    dl.phase_count = static_cast<int32_t>(lp.phases.size());
    dl.span_bytes = lp.span_bytes;
    // The resident ranges by offset: the arena's order.
    std::vector<const Range*> by_offset;
    for (const Range& r : lp.ranges)
      if (r.source == Source::Resident) by_offset.push_back(&r);
    std::sort(by_offset.begin(), by_offset.end(), [](const Range* a, const Range* b) { return a->offset < b->offset; });
    for (size_t ph = 0; ph < lp.phases.size(); ++ph) {
      const Phase& phase = lp.phases[ph];
      DevPhase d;
      d.kind = phase.kind == PhaseKind::Stream ? 1 : 0;
      d.resolves = phase.resolves < 0 ? 0xff : static_cast<uint8_t>(phase.resolves);
      d.latency_us = phase.latency_us;
      d.fold_after_us = phase.fold_after_us;
      d.seg_begin = static_cast<int32_t>(dp.segments.size());
      d.lead_begin = lead;
      d.layer = static_cast<int32_t>(dp.layers.size());
      for (const Range& r : lp.ranges)
        if (r.phase == static_cast<int>(ph) && r.source == Source::Host && r.reads > 0) d.waits_host = 1;
      if (consumers && phase.name == "accumulate") { d.accumulate = 1; d.kind = 1; }
      // The shared expert's fp8 pair or down and the fp8 projections as
      // consumer pools of one group: their ranges leave the fixed segments
      // and become lead-only segments.
      std::vector<const Range*> shared_pool;
      uint32_t phase_part_elems = 0;
      if (consumers) {
        for (size_t ri = 0; ri < lp.ranges.size(); ++ri) {
          const Range& r = lp.ranges[ri];
          if (r.phase != static_cast<int>(ph) || r.dep != Dependency::Fixed || r.source != Source::Resident || r.reads == 0) continue;
          if (r.format != Format::Fp8E4m3 || (r.consumer != Consumer::Projection && r.consumer != Consumer::Indexer)) continue;
          const Range* scales = nullptr;
          for (const Range& c : lp.ranges)
            if (c.scale_of == static_cast<int>(ri)) scales = &c;
          if (!scales) continue;
          DevPool pool;
          pool.kind = kPoolFp8Matrix;
          pool.group_kind = kGroupTokens;
          pool.groups = 1;
          pool.stride = 0;
          pool.base = std::min(r.offset, scales->offset);
          pool.bytes = std::max(align_grant(r.offset + r.bytes), align_grant(scales->offset + scales->bytes)) - pool.base;
          pool.offsets[0] = r.offset - pool.base;
          pool.offsets[1] = scales->offset - pool.base;
          pool.rows = static_cast<int32_t>(r.rows);
          pool.k = static_cast<int32_t>(r.cols);
          pool.slabs = projection_slabs(pool.k);
          pool.unit_rows = kProjectionUnitRows;
          pool.units_per_group = ((pool.rows + pool.unit_rows - 1) / pool.unit_rows) * pool.slabs;  // the reduce units follow at run time
          pool.grab = 1;
          pool.out_offset = dp.proj_elems;
          dp.proj_elems += static_cast<uint32_t>(pool.rows) * kStepRows;
          if (pool.slabs > 1) {
            pool.part_offset = phase_part_elems;
            phase_part_elems += static_cast<uint32_t>(pool.slabs) * static_cast<uint32_t>(pool.rows) * kStepRows;
          }
          dp.fixed_bytes += static_cast<double>(r.bytes_per_read()) * r.reads + static_cast<double>(scales->bytes_per_read()) * scales->reads;
          add_pool(dp, d, pool);
          shared_pool.push_back(&r);
          shared_pool.push_back(scales);
        }
        const Range* payload[3] = {nullptr, nullptr, nullptr};
        const Range* scales[3] = {nullptr, nullptr, nullptr};
        for (const Range* r : by_offset) {
          if (r->phase != static_cast<int>(ph) || r->dep != Dependency::Fixed || r->matrix < 0 || r->matrix > 2) continue;
          if (r->consumer != Consumer::SharedGateUp && r->consumer != Consumer::SharedDown) continue;
          if (r->format == Format::Fp8E4m3) payload[r->matrix] = r;
          else if (r->format == Format::ScaleF32) scales[r->matrix] = r;
        }
        const bool gate_up = payload[0] && scales[0] && payload[1] && scales[1] && !payload[2];
        const bool down = payload[2] && scales[2] && !payload[0];
        if (gate_up || down) {
          DevPool pool;
          pool.kind = gate_up ? kPoolFp8GateUp : kPoolFp8Down;
          pool.groups = 1;
          pool.stride = 0;
          const Range* parts[4] = {gate_up ? payload[0] : payload[2], gate_up ? scales[0] : scales[2], gate_up ? payload[1] : nullptr,
                                   gate_up ? scales[1] : nullptr};
          uint64_t lo = UINT64_MAX, hi = 0;
          for (const Range* r : parts)
            if (r) { lo = std::min(lo, r->offset); hi = std::max(hi, align_grant(r->offset + r->bytes)); shared_pool.push_back(r); }
          pool.base = lo;
          pool.bytes = hi - lo;
          for (int i = 0; i < 4; ++i) pool.offsets[i] = parts[i] ? parts[i]->offset - lo : 0;
          pool.rows = static_cast<int32_t>(parts[0]->rows);
          pool.k = static_cast<int32_t>(parts[0]->cols);
          pool.units_per_group = (pool.rows + 127) / 128;
          pool.grab = down ? 4 : 1;
          pool.group_kind = kGroupShared;
          for (const Range* r : parts)
            if (r) dp.fixed_bytes += static_cast<double>(r->bytes_per_read()) * r->reads;
          add_pool(dp, d, pool);
        }
      }
      // Fixed-class ranges of this phase, coalesced where adjacent.
      uint32_t units = 0;
      for (const Range* r : by_offset) {
        if (r->phase != static_cast<int>(ph) || r->dep == Dependency::Router || r->reads == 0) continue;
        if (std::find(shared_pool.begin(), shared_pool.end(), r) != shared_pool.end()) continue;
        const uint64_t end = align_grant(r->offset + r->bytes);
        if (!dp.segments.empty() && static_cast<int32_t>(dp.segments.size()) > d.seg_begin &&
            align_grant(dp.segments.back().offset + dp.segments.back().bytes) == r->offset) {
          dp.segments.back().bytes = end - dp.segments.back().offset;
        } else {
          DevSegment s;
          s.offset = r->offset;
          s.bytes = end - r->offset;
          dp.segments.push_back(s);
        }
        dp.fixed_bytes += static_cast<double>(r->bytes_per_read()) * r->reads;
      }
      for (int32_t i = d.seg_begin; i < static_cast<int32_t>(dp.segments.size()); ++i) {
        DevSegment& s = dp.segments[static_cast<size_t>(i)];
        s.unit_begin = units;
        s.units = static_cast<uint32_t>((s.bytes + unit_bytes - 1) / unit_bytes);
        units += s.units;
      }
      d.seg_count = static_cast<int32_t>(dp.segments.size()) - d.seg_begin;
      d.fixed_units = units;
      // The shared pool's bytes as lead-only segments, numbered after the fixed units.
      d.lead_seg_begin = static_cast<int32_t>(dp.segments.size());
      uint32_t lead_units = units;
      if (!shared_pool.empty()) {
        std::vector<const Range*> sorted = shared_pool;
        std::sort(sorted.begin(), sorted.end(), [](const Range* a, const Range* b) { return a->offset < b->offset; });
        for (const Range* r : sorted) {
          const uint64_t end = align_grant(r->offset + r->bytes);
          if (static_cast<int32_t>(dp.segments.size()) > d.lead_seg_begin &&
              align_grant(dp.segments.back().offset + dp.segments.back().bytes) == r->offset) {
            dp.segments.back().bytes = end - dp.segments.back().offset;
          } else {
            DevSegment s;
            s.offset = r->offset;
            s.bytes = end - r->offset;
            dp.segments.push_back(s);
          }
        }
        for (int32_t i = d.lead_seg_begin; i < static_cast<int32_t>(dp.segments.size()); ++i) {
          DevSegment& s = dp.segments[static_cast<size_t>(i)];
          s.unit_begin = lead_units;
          s.units = static_cast<uint32_t>((s.bytes + unit_bytes - 1) / unit_bytes);
          lead_units += s.units;
        }
      }
      d.lead_seg_count = static_cast<int32_t>(dp.segments.size()) - d.lead_seg_begin;
      d.lead_units = lead_units;
      lead += lead_units;
      dp.part_elems = std::max(dp.part_elems, phase_part_elems);
      // The router-bound pool: every group's ranges at one stride.
      std::vector<const Range*> routed;
      for (const Range* r : by_offset)
        if (r->phase == static_cast<int>(ph) && r->dep == Dependency::Router) routed.push_back(r);
      if (!routed.empty()) {
        if (lp.groups <= 0) throw std::logic_error("lower_plan: router-bound ranges in a layer without groups");
        std::vector<uint64_t> first(static_cast<size_t>(lp.groups), UINT64_MAX), last(static_cast<size_t>(lp.groups), 0);
        for (const Range* r : routed) {
          auto& f = first[static_cast<size_t>(r->group)];
          auto& l = last[static_cast<size_t>(r->group)];
          f = std::min(f, r->offset);
          l = std::max(l, align_grant(r->offset + r->bytes));
          dp.router_bytes += static_cast<double>(r->bytes_per_read()) * r->reads * (lp.expected_groups / lp.groups);
        }
        DevPool pool;
        pool.base = first[0];
        pool.bytes = last[0] - first[0];
        pool.groups = lp.groups;
        pool.stride = lp.groups > 1 ? first[1] - first[0] : pool.bytes;
        for (int g = 0; g < lp.groups; ++g) {
          const size_t gi = static_cast<size_t>(g);
          if (first[gi] != pool.base + static_cast<uint64_t>(g) * pool.stride || last[gi] - first[gi] != pool.bytes)
            throw std::logic_error("lower_plan: layer " + std::to_string(lp.layer) + " phase " + phase.name +
                                   ": the routed groups are not at one stride (group " + std::to_string(g) + ")");
        }
        pool.units_per_group = static_cast<int32_t>((pool.bytes + unit_bytes - 1) / unit_bytes);
        // The MXFP4 expert consumers: group 0's matrices in the arena's order
        // (payload then scales per matrix), units of 128 output rows.
        if (consumers) {
          const Range* payload[3] = {nullptr, nullptr, nullptr};
          const Range* scales[3] = {nullptr, nullptr, nullptr};
          for (const Range* r : routed)
            if (r->group == 0 && r->matrix >= 0 && r->matrix < 3 && r->format == Format::Fp4E2m1) payload[r->matrix] = r;
            else if (r->group == 0 && r->matrix >= 0 && r->matrix < 3 && r->format == Format::ScaleE8m0) scales[r->matrix] = r;
          const bool gate_up = payload[0] && scales[0] && payload[1] && scales[1] && !payload[2];
          const bool down = payload[2] && scales[2] && !payload[0];
          if (gate_up || down) {
            const Range* m0 = gate_up ? payload[0] : payload[2];
            // The layout the plan records picks the form: the tile-major form
            // where the arena wrote tiles, the bridged row-major form otherwise.
            const bool tiled = m0->layout == Layout::Tile16;
            pool.kind = gate_up ? (tiled ? kPoolMxfp4GateUp : kPoolMxfp4RowGateUp) : (tiled ? kPoolMxfp4Down : kPoolMxfp4RowDown);
            pool.rows = static_cast<int32_t>(m0->rows);
            pool.k = static_cast<int32_t>(m0->cols);
            if (gate_up) {
              pool.offsets[0] = payload[0]->offset - pool.base;
              pool.offsets[1] = scales[0]->offset - pool.base;
              pool.offsets[2] = payload[1]->offset - pool.base;
              pool.offsets[3] = scales[1]->offset - pool.base;
            } else {
              pool.offsets[0] = payload[2]->offset - pool.base;
              pool.offsets[1] = scales[2]->offset - pool.base;
            }
            pool.units_per_group = (pool.rows + 127) / 128;
            pool.grab = down ? 4 : 1;  // the down units are small (phase 1 measured four a grab)
          }
        }
        pool.group_kind = kGroupRouted;
        add_pool(dp, d, pool);
      }
      dp.phases.push_back(d);
    }
    dp.layers.push_back(dl);
  }
  dp.lead_units = lead;
  // The expert phases: a gate and up pool followed by a down pool merge, and
  // each down pool names the gate and up pool of its group kind as partner.
  if (consumers)
    for (size_t i = 0; i + 1 < dp.phases.size(); ++i) {
      DevPhase& a = dp.phases[i];
      DevPhase& b = dp.phases[i + 1];
      if (a.routed < 0 || b.routed < 0 || a.layer != b.layer) continue;
      if (!pool_is_gate_up(dp.pools[static_cast<size_t>(a.pools[a.routed])].kind) ||
          !pool_is_down(dp.pools[static_cast<size_t>(b.pools[b.routed])].kind))
        continue;
      a.merged_next = 1;
      for (int bi = 0; bi < b.n_pools; ++bi) {
        DevPool& down = dp.pools[static_cast<size_t>(b.pools[bi])];
        if (!pool_is_down(down.kind)) continue;
        for (int ai = 0; ai < a.n_pools; ++ai) {
          const DevPool& gate = dp.pools[static_cast<size_t>(a.pools[ai])];
          if (pool_is_gate_up(gate.kind) && gate.group_kind == down.group_kind) down.partner = a.pools[ai];
        }
        if (down.partner < 0) throw std::logic_error("lower_plan: a down pool without its gate and up pool");
      }
    }
  return dp;
}

std::string device_plan_summary(const DevicePlan& dp) {
  char line[256];
  std::string out;
  std::snprintf(line, sizeof line, "%zu layers, %zu phases, %zu segments, %zu pools, unit %u B, %u lead units; step bytes fixed %.3f GB, routed %.3f GB\n",
                dp.layers.size(), dp.phases.size(), dp.segments.size(), dp.pools.size(), dp.unit_bytes, dp.lead_units,
                dp.fixed_bytes / 1e9, dp.router_bytes / 1e9);
  out += line;
  return out;
}

}  // namespace dgpp::step
