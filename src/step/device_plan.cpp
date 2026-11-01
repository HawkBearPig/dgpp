// The stream plan's lowering (step/device_plan.hpp).
#include "step/device_plan.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

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
  if (pool.kind != kPoolMhc) dp.max_k = std::max(dp.max_k, pool.k);  // a hyper-connection pool's width is the streams', not an activation row's
  return index;
}

}  // namespace

DevicePlan lower_plan(const std::vector<LayerPlan>& layers, uint32_t unit_bytes, unsigned consumer_mask) {
  if (unit_bytes == 0 || unit_bytes % 512 != 0) throw std::invalid_argument("lower_plan: unit_bytes must be a multiple of 512");
  const bool experts = (consumer_mask & kConsumeExperts) != 0, shared = (consumer_mask & kConsumeShared) != 0,
             projections = (consumer_mask & kConsumeProjections) != 0, router = (consumer_mask & kConsumeRouter) != 0,
             mhc = (consumer_mask & kConsumeMhc) != 0, attention = (consumer_mask & kConsumeAttention) != 0,
             chain = (consumer_mask & kConsumeChain) != 0;
  const bool consumers = consumer_mask != 0;
  DevicePlan dp;
  dp.unit_bytes = unit_bytes;
  uint32_t lead = 0;
  // The pools by the range they were lowered from, for the chain's wiring (a layer's pools only).
  std::vector<std::string> pool_names;
  auto name_pool = [&](const std::string& name) {
    pool_names.resize(dp.pools.size());
    pool_names.back() = name;
  };
  size_t layer_pool_begin = 0;
  auto find_pool = [&](const char* name) -> int {
    for (size_t i = layer_pool_begin; i < pool_names.size(); ++i)
      if (pool_names[i] == name) return static_cast<int>(i);
    return -1;
  };
  for (const LayerPlan& lp : layers) {
    layer_pool_begin = dp.pools.size();
    DevLayer dl;
    dl.phase_begin = static_cast<int32_t>(dp.phases.size());
    dl.phase_count = static_cast<int32_t>(lp.phases.size());
    dl.span_bytes = lp.span_bytes;
    // The resident ranges by offset: the arena's order.
    std::vector<const Range*> by_offset;
    for (const Range& r : lp.ranges)
      if (r.source == Source::Resident) by_offset.push_back(&r);
    std::sort(by_offset.begin(), by_offset.end(), [](const Range* a, const Range* b) { return a->offset < b->offset; });
    int32_t mhc_hidden = 0;  // the sites' hidden width once the attention site's pool is lowered
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
      if (experts && phase.name == "accumulate") { d.accumulate = 1; d.kind = 1; if (mhc) d.update = 1; }
      // The shared expert's fp8 pair or down and the fp8 projections as
      // consumer pools of one group: their ranges leave the fixed segments
      // and become lead-only segments.
      std::vector<const Range*> shared_pool;
      uint32_t phase_part_elems = 0;
      if (consumers) {
        for (size_t ri = 0; projections && ri < lp.ranges.size(); ++ri) {
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
          pool.grab = 1;  // (two a grab measured 1.5 points worse and noisier on 2026-10-31: the fewer, longer grabs ragged the tail)
          pool.out_offset = dp.proj_elems;
          dp.proj_elems += static_cast<uint32_t>(pool.rows) * kStepRows;
          if (pool.slabs > 1) {
            pool.part_offset = phase_part_elems;
            phase_part_elems += static_cast<uint32_t>(pool.slabs) * static_cast<uint32_t>(pool.rows) * kStepRows;
          }
          dp.fixed_bytes += static_cast<double>(r.bytes_per_read()) * r.reads + static_cast<double>(scales->bytes_per_read()) * scales->reads;
          // The attention site's stream update rides on the output projection's reduce units,
          // which hold its final y for their chunk: no phase of its own, no round trip.
          if (mhc && r.name == "attn.wo_b.payload") {
            if (mhc_hidden == 0 || pool.rows != mhc_hidden)
              throw std::logic_error("lower_plan: the attention update needs the hyper-connection site before the output projection, at its hidden width");
            if (pool.slabs < 2) throw std::logic_error("lower_plan: the attention update rides on the output projection's reduce units, which need two slabs or more");
            pool.update_site = 1;
          }
          add_pool(dp, d, pool);
          name_pool(r.name);
          shared_pool.push_back(&r);
          shared_pool.push_back(scales);
        }
        // The projection chain's wiring (2026-11-01): the data flow of models/dsv41/csa2_layer.cu
        // (project_q_kv, the decode tail, project_out). A dependent pool names the pool whose y
        // it reads (in_pool) and its class opens on that pool's completion. Behind wq_a and wkv
        // the row tails: the q norm, the kv norm with its rotation and ring append, the q and
        // index-q rotations, each a pool of one unit a row writing its own y, so every stage's
        // output stays for the production kernels to check it against.
        if (projections && chain) {
          auto find_range = [&](const char* name) -> const Range* {
            for (const Range& c : lp.ranges)
              if (c.phase == static_cast<int>(ph) && c.source == Source::Resident && c.consumer == Consumer::Norm && c.name == name) return &c;
            return nullptr;
          };
          auto chain = [&](int pool, int src) {
            DevPool& q = dp.pools[static_cast<size_t>(pool)];
            if (q.k != dp.pools[static_cast<size_t>(src)].rows)
              throw std::logic_error("lower_plan: " + pool_names[static_cast<size_t>(pool)] + " reads " + pool_names[static_cast<size_t>(src)] +
                                     "'s y, whose width is not its inner width");
            q.in_src = 1;
            q.in_pool = src;
            q.gate_kind = kGatePool;
          };
          auto tail_pool = [&](const std::string& name, const Range* norm, int src, int rope_heads, int rope_head_stride, int rope_off, bool append,
                               int gate_pool) -> int {
            const DevPool& sp = dp.pools[static_cast<size_t>(src)];
            DevPool pool;
            pool.kind = kPoolRowTail;
            pool.group_kind = kGroupTokens;
            pool.groups = 1;
            pool.stride = 0;
            if (norm) {
              if (norm->cols != sp.rows)
                throw std::logic_error("lower_plan: the norm " + norm->name + " is not the width of " + pool_names[static_cast<size_t>(src)] + "'s rows");
              pool.base = norm->offset;
              pool.bytes = align_grant(norm->offset + norm->bytes) - norm->offset;
              pool.offsets[0] = 0;
              pool.k = static_cast<int32_t>(norm->cols);
              dp.fixed_bytes += static_cast<double>(norm->bytes_per_read()) * norm->reads;
              shared_pool.push_back(norm);
            }
            if (rope_heads && rope_heads * rope_head_stride != sp.rows)
              throw std::logic_error("lower_plan: the rotation of " + name + " does not cover its source's rows");
            if (append && sp.rows != kAttnLatent) throw std::logic_error("lower_plan: the appended kv row of " + name + " is not the latent width");
            pool.rows = sp.rows;  // the row width: its source's
            pool.units_per_group = 1;  // a row a unit; the kernel multiplies by the step's rows
            // A slabbed source's partials are summed by the tail's units (a row each, in slab order),
            // not by a reduce class between the two: one dependent stage fewer on the chain.
            if (sp.kind == kPoolFp8Matrix && sp.slabs > 1) dp.pools[static_cast<size_t>(src)].reduce_by_tail = 1;
            pool.grab = 1;
            pool.in_src = 1;
            pool.in_pool = src;
            pool.gate_pool = gate_pool;
            pool.gate_kind = kGatePool;
            pool.rope_heads = rope_heads;
            pool.rope_head_stride = rope_head_stride;
            pool.rope_off = rope_off;
            pool.append = append ? 1 : 0;
            pool.out_offset = dp.proj_elems;
            dp.proj_elems += static_cast<uint32_t>(pool.rows) * kStepRows;
            const int index = add_pool(dp, d, pool);
            name_pool(name);
            return index;
          };
          const int wq_a = find_pool("attn.wq_a.payload"), wq_b = find_pool("attn.wq_b.payload"), wkv = find_pool("attn.wkv.payload");
          const int idx_wq_b = find_pool("attn.indexer.wq_b.payload");
          const Range* q_norm = find_range("attn.q_norm");
          const Range* kv_norm = find_range("attn.kv_norm");
          // The q projections rotate their heads in the unit's epilogue (the decode tail's q rope and
          // the indexer's: a class of its own at the phase's end cost 12-17 µs a layer, 2026-11-01).
          auto epilogue_rope = [&](int pool, int head) {
            DevPool& q = dp.pools[static_cast<size_t>(pool)];
            if (q.rows % head != 0) throw std::logic_error("lower_plan: " + pool_names[static_cast<size_t>(pool)] + "'s rows are not whole heads");
            if (q.slabs != 1) throw std::logic_error("lower_plan: the epilogue rotation of " + pool_names[static_cast<size_t>(pool)] + " needs the whole row in the unit (one slab)");
            q.rope_heads = q.rows / head;
            q.rope_head_stride = head;
            q.rope_off = head - kAttnRope;
          };
          if (wq_a >= 0 && wq_b >= 0 && wkv >= 0 && q_norm && kv_norm) {
            const int qn = tail_pool("attn.q_norm", q_norm, wq_a, 0, 0, 0, false, -1);
            tail_pool("attn.kv_tail", kv_norm, wkv, 1, kAttnLatent, kAttnLatent - kAttnRope, true, -1);
            chain(wq_b, qn);
            epilogue_rope(wq_b, kAttnLatent);
            if (idx_wq_b >= 0) {
              chain(idx_wq_b, qn);
              epilogue_rope(idx_wq_b, kAttnIndexDim);
            }
          }
          // The output projections: wo_a reads the attention's o block-diagonally (group g's rows
          // read the heads of group g), wo_b reads wo_a's y. The attention's units are the group
          // before, so o is complete when the phase opens; wo_b's class opens on wo_a.
          const int wo_a = find_pool("attn.wo_a.payload"), wo_b = find_pool("attn.wo_b.payload");
          int attn_heads = 0;
          for (size_t i = layer_pool_begin; i < dp.pools.size(); ++i)
            if (dp.pools[i].kind == kPoolAttention) attn_heads = dp.pools[i].unit_rows;
          if (wo_a >= 0 && attn_heads > 0) {
            DevPool& q = dp.pools[static_cast<size_t>(wo_a)];
            const int o_width = attn_heads * kAttnLatent;
            if (q.k <= 0 || o_width % q.k != 0) throw std::logic_error("lower_plan: wo_a's inner width does not divide the attention output's");
            const int groups = o_width / q.k;
            if (q.rows % groups != 0 || (q.rows / groups) % kProjectionUnitRows != 0)
              throw std::logic_error("lower_plan: wo_a's rows do not cut into whole unit blocks a head group");
            q.in_src = 2;
            q.in_group_rows = q.rows / groups;
            q.in_group_cols = q.k;
          }
          if (wo_a >= 0 && wo_b >= 0) chain(wo_b, wo_a);
        }
        // A hyper-connection site as a consumer pool: the coefficient matrix, its
        // base and scale, and the sublayer's norm weight; a unit is one
        // (coefficient, token) dot and the token's last unit runs the finish.
        if (mhc) {
          const Range* fn = nullptr;
          const Range* hbase = nullptr;
          const Range* hscale = nullptr;
          const Range* norm = nullptr;
          for (const Range* r : by_offset) {
            if (r->phase != static_cast<int>(ph) || r->source != Source::Resident || r->reads == 0) continue;
            if (r->consumer == Consumer::HyperConn && r->format == Format::Bf16) fn = r;
            else if (r->consumer == Consumer::HyperConn && r->format == Format::F32 && r->bytes == 24 * 4) hbase = r;
            else if (r->consumer == Consumer::HyperConn && r->format == Format::F32 && r->bytes == 3 * 4) hscale = r;
            else if (r->consumer == Consumer::Norm && r->format == Format::Bf16) norm = r;
          }
          if (fn && hbase && hscale && norm && fn->rows == 24 && fn->rows % kMhcUnitCoeffs == 0) {
            DevPool pool;
            pool.kind = kPoolMhc;
            pool.group_kind = kGroupTokens;
            pool.groups = 1;
            pool.stride = 0;
            const Range* parts[4] = {fn, hbase, hscale, norm};
            uint64_t lo = UINT64_MAX, hi = 0;
            for (const Range* r : parts) { lo = std::min(lo, r->offset); hi = std::max(hi, align_grant(r->offset + r->bytes)); }
            pool.base = lo;
            pool.bytes = hi - lo;
            for (int i = 0; i < 4; ++i) pool.offsets[i] = parts[i]->offset - lo;
            pool.rows = static_cast<int32_t>(fn->rows);   // the coefficients
            pool.k = static_cast<int32_t>(fn->cols);      // 4 hidden
            pool.unit_rows = kMhcUnitCoeffs;
            pool.units_per_group = pool.rows / kMhcUnitCoeffs;  // a token's units; the kernel multiplies by the step's rows
            pool.grab = 1;
            pool.site = phase.name == "ffn_in" ? 1 : 0;
            if (pool.site == 0) mhc_hidden = pool.k / 4;
            for (const Range* r : parts) dp.fixed_bytes += static_cast<double>(r->bytes_per_read()) * r->reads;
            add_pool(dp, d, pool);
            for (const Range* r : parts) shared_pool.push_back(r);
            // Real work in place of the modeled latency; the plan's latency stays as the phase's
            // idle allowance for the lead (the units are latency-bound, the memory free).
            d.kind = 1;
          }
        }
        // The router as a consumer pool: its bf16 matrix and F32 bias, units of
        // 16 expert rows, the scores of every token out; the chain selects.
        if (router) {
          const Range* weights = nullptr;
          const Range* bias = nullptr;
          for (const Range* r : by_offset) {
            if (r->phase != static_cast<int>(ph) || r->consumer != Consumer::Router || r->source != Source::Resident || r->reads == 0) continue;
            if (r->format == Format::Bf16 && r->rows > 0 && r->cols > 0) weights = r;
            else if (r->format == Format::F32) bias = r;
          }
          if (weights && bias && weights->rows % 16 == 0) {
            DevPool pool;
            pool.kind = kPoolRouter;
            pool.group_kind = kGroupTokens;
            pool.groups = 1;
            pool.stride = 0;
            pool.base = std::min(weights->offset, bias->offset);
            pool.bytes = std::max(align_grant(weights->offset + weights->bytes), align_grant(bias->offset + bias->bytes)) - pool.base;
            pool.offsets[0] = weights->offset - pool.base;
            pool.offsets[1] = bias->offset - pool.base;
            pool.rows = static_cast<int32_t>(weights->rows);
            pool.k = static_cast<int32_t>(weights->cols);
            pool.unit_rows = 16;  // two expert rows a warp, the activations read once for both
            pool.units_per_group = pool.rows / 16;
            pool.grab = 1;
            dp.fixed_bytes += static_cast<double>(weights->bytes_per_read()) * weights->reads + static_cast<double>(bias->bytes_per_read()) * bias->reads;
            dp.n_experts = std::max(dp.n_experts, pool.rows);
            add_pool(dp, d, pool);
            shared_pool.push_back(weights);
            shared_pool.push_back(bias);
          }
        }
        // The CSA2 attention as consumer pools, one per latent source the phase reads from the
        // state pool (the window ring's fp8_block rows, the main cache's fp4_block selected
        // entries): a unit is one block of the production flash kernel, 32 (row, head) M-rows
        // over one split of 16 keys (kernels/dsa_attn_flash.cuh), so a source's splits are its
        // keys a read over the tile; the phase's last attention pool carries the finish as its
        // reduce class, one (row, head) a unit, merging every source's group partials with the
        // layer's sink (its attn_sink range) and the rotary table (kernels/csa2_attn_finish.cuh).
        // The pools' bytes are the state pool's, not the arena's: no segment, no lead.
        if (attention) {
          const Range* sink = nullptr;
          for (const Range& r : lp.ranges)
            if (r.source == Source::Resident && r.consumer == Consumer::Attention && r.format == Format::F32 && r.name == "attn.attn_sink") sink = &r;
          int last_attention = -1;
          for (const Range& r : lp.ranges) {
            if (r.phase != static_cast<int>(ph) || r.source != Source::Pool || r.consumer != Consumer::Attention || r.reads == 0) continue;
            if (r.format != Format::KvFp8Block && r.format != Format::KvFp4Block) continue;
            if (!sink) throw std::logic_error("lower_plan: the attention consumer needs the layer's attn_sink range");
            if (r.rows <= 0 || r.rows % kAttnTile != 0)
              throw std::logic_error("lower_plan: an attention source's keys a read (" + r.name + ") must be whole 16-key tiles");
            DevPool pool;
            pool.kind = kPoolAttention;
            pool.group_kind = kGroupTokens;
            pool.groups = 1;
            pool.stride = 0;
            pool.base = 0;
            pool.bytes = 0;
            pool.site = r.format == Format::KvFp8Block ? 0 : 1;
            pool.rows = static_cast<int32_t>(r.rows / kAttnTile);
            pool.k = kAttnLatent;
            pool.slabs = pool.rows % kAttnGroup == 0 ? kAttnGroup : 1;
            pool.unit_rows = static_cast<int32_t>(sink->cols);  // local_heads
            pool.units_per_group = pool.rows;  // times the step's (row, head) blocks, at run time
            pool.grab = 1;
            pool.offsets[0] = sink->offset;
            last_attention = d.n_pools;
            add_pool(dp, d, pool);
            name_pool(r.name);
          }
          if (last_attention >= 0) dp.pools[static_cast<size_t>(d.pools[last_attention])].gate_arg = 1;
          // The queries from the chain: wq_b's y, rotated in its units' epilogue; the class opens on
          // the kv tail too (the ring holds the step's rows and their slot lists when the attention
          // reads them).
          const int wq_b = find_pool("attn.wq_b.payload"), kv_tail = find_pool("attn.kv_tail");
          if (wq_b >= 0 && kv_tail >= 0 && last_attention >= 0 && dp.pools[static_cast<size_t>(wq_b)].rope_heads > 0) {
            if (dp.pools[static_cast<size_t>(wq_b)].rows != static_cast<int32_t>(sink->cols) * kAttnLatent)
              throw std::logic_error("lower_plan: the q rows are not the attention's local heads by the latent width");
            for (int i = 0; i < d.n_pools; ++i) {
              DevPool& q = dp.pools[static_cast<size_t>(d.pools[i])];
              if (q.kind != kPoolAttention) continue;
              q.in_src = 1;
              q.in_pool = wq_b;
              q.gate_pool = kv_tail;
              q.gate_kind = kGatePool;
            }
          }
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
        if (shared && (gate_up || down)) {
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
          pool.grab = down ? 8 : 1;
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
      // A gated pool's bytes as lead units: the units of the lead-only segments that cover its span
      // (unit boundaries may take a neighbour's bytes; the units are numbered across the segments).
      for (int i = 0; i < d.n_pools; ++i) {
        DevPool& q = dp.pools[static_cast<size_t>(d.pools[i])];
        if (q.gate_kind != kGatePool || q.kind != kPoolFp8Matrix || q.bytes == 0) continue;
        uint32_t first = UINT32_MAX, last = 0;
        for (int32_t si = d.lead_seg_begin; si < static_cast<int32_t>(dp.segments.size()); ++si) {
          const DevSegment& sg = dp.segments[static_cast<size_t>(si)];
          const uint64_t lo = std::max(sg.offset, q.base), hi = std::min(sg.offset + sg.bytes, q.base + q.bytes);
          if (lo >= hi) continue;
          first = std::min<uint32_t>(first, sg.unit_begin + static_cast<uint32_t>((lo - sg.offset) / unit_bytes));
          last = std::max<uint32_t>(last, sg.unit_begin + static_cast<uint32_t>((hi - 1 - sg.offset) / unit_bytes) + 1);
        }
        if (first == UINT32_MAX) throw std::logic_error("lower_plan: a gated pool's bytes are not in its phase's lead-only segments");
        q.stage_unit_begin = first;
        q.stage_units = last - first;
      }
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
        if (experts) {
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
            pool.grab = down ? 8 : 1;  // the down units are small (phase 1 measured four a grab)
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
  // The merged groups. The hyper-connection sites and the router join the
  // stream phase before them as gated classes: the MoE site and the router
  // inside attn_out, behind its fold (triggered by the output projection's
  // reduce units, which carry the attention update); the attention site
  // inside the previous layer's accumulate, behind its fold. Measured as
  // phases of their own (2026-10-31) each cost a boundary of about 5 µs and
  // ran latency-bound while the memory idled; as classes they fill the
  // stream phase's drain. A latency phase (no pool) is never merged.
  auto has_pool = [&](const DevPhase& d, uint8_t kind, int site) {
    for (int i = 0; i < d.n_pools; ++i) {
      const DevPool& pool = dp.pools[static_cast<size_t>(d.pools[i])];
      if (pool.kind == kind && (site < 0 || pool.site == site)) return true;
    }
    return false;
  };
  if (mhc)
    for (size_t i = 0; i + 1 < dp.phases.size(); ++i) {
      DevPhase& a = dp.phases[i];
      if (a.kind != 1 || a.merged) continue;
      // attn_out: its update-carrying projection first in the hand-out, the fold on its reduce.
      int update_pool = -1;
      for (int k = 0; k < a.n_pools; ++k)
        if (dp.pools[static_cast<size_t>(a.pools[k])].update_site) update_pool = k;
      if (update_pool >= 0 && a.fold_after_us > 0.f) {
        std::swap(a.pools[0], a.pools[update_pool]);
        if (a.routed == 0) a.routed = static_cast<int8_t>(update_pool);
        else if (a.routed == update_pool) a.routed = 0;
        a.fold_trigger = kTriggerReduce;
        a.trigger_pool = a.pools[0];
        DevPhase& b = dp.phases[i + 1];
        if (b.layer == a.layer && b.kind == 1 && has_pool(b, kPoolMhc, 1)) {
          a.merged = 1;
          for (int k = 0; k < b.n_pools; ++k) {
            DevPool& pool = dp.pools[static_cast<size_t>(b.pools[k])];
            if (pool.kind == kPoolMhc) pool.gate_kind = kGateFold;
          }
          if (i + 2 < dp.phases.size()) {
            DevPhase& c = dp.phases[i + 2];
            if (c.layer == a.layer && c.kind == 1 && has_pool(c, kPoolRouter, -1)) {
              a.merged = 2;
              for (int k = 0; k < c.n_pools; ++k) {
                DevPool& pool = dp.pools[static_cast<size_t>(c.pools[k])];
                if (pool.kind == kPoolRouter) { pool.gate_kind = kGateSite; pool.gate_arg = 1; }
              }
            }
          }
        }
      }
      // accumulate: the next layer's attention site behind the layer's fold.
      if (a.accumulate && a.fold_after_us > 0.f) {
        a.fold_trigger = kTriggerAccumulate;
        DevPhase& b = dp.phases[i + 1];
        if (b.layer == a.layer + 1 && b.kind == 1 && b.n_pools == 1 && b.fixed_units == 0 && !b.waits_host && has_pool(b, kPoolMhc, 0)) {
          a.merged = 1;
          dp.pools[static_cast<size_t>(b.pools[0])].gate_kind = kGateFold;
        }
      }
    }
  // The last phase of the plan is an accumulate with a fold too (the loop above stops one short
  // of it, having nothing to merge behind it): its trigger is its own units, so the step engine's
  // fold carries it as every other layer's (2026-11-01: the last layer's MoE fold was the modeled
  // spin and its update went missing past world 1).
  if (mhc && !dp.phases.empty()) {
    DevPhase& z = dp.phases.back();
    if (z.accumulate && z.fold_after_us > 0.f && z.fold_trigger == kTriggerEnd) z.fold_trigger = kTriggerAccumulate;
  }
  // The attention behind the stream phase before it, where they are consecutive (the layers
  // without an index source; a selection latency phase between them keeps the attention a phase
  // of its own until the selection is units too): its classes follow the projections' in the
  // hand-out, so they run in the projections' drain, and the output projection publishes when
  // the group ends, after the finish. Measured as a phase of its own (2026-11-01) the attention
  // cost the layer six points. Gated on nothing yet: the queries are given, not produced here.
  if (attention)
    for (size_t i = 0; i + 1 < dp.phases.size(); ++i) {
      DevPhase& a = dp.phases[i];
      DevPhase& b = dp.phases[i + 1];
      if (a.kind != 1 || a.merged || a.accumulate || a.fold_after_us > 0.f || b.layer != a.layer || b.kind != 1 || b.merged) continue;
      if (b.n_pools == 0 || b.fixed_units != 0 || b.waits_host) continue;
      bool all_attention = true;
      for (int k = 0; k < b.n_pools; ++k)
        if (dp.pools[static_cast<size_t>(b.pools[k])].kind != kPoolAttention) all_attention = false;
      if (!all_attention) continue;
      a.merged = 1;
    }
  // The expert phases: a gate and up pool followed by a down pool merge, and
  // each down pool names the gate and up pool of its group kind as partner.
  if (experts || shared)
    for (size_t i = 0; i + 1 < dp.phases.size(); ++i) {
      DevPhase& a = dp.phases[i];
      DevPhase& b = dp.phases[i + 1];
      if (a.routed < 0 || b.routed < 0 || a.layer != b.layer) continue;
      if (!pool_is_gate_up(dp.pools[static_cast<size_t>(a.pools[a.routed])].kind) ||
          !pool_is_down(dp.pools[static_cast<size_t>(b.pools[b.routed])].kind))
        continue;
      a.merged = 1;
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
  // The groups' facts for the chain (device_plan.hpp DevPhase::group_*), after the merges, and
  // the step's fold order: every group head with a fold the kernel can carry (a trigger inside
  // the group) takes the next fold index; the host issues that many generations a step.
  size_t member_until = 0;
  for (size_t i = 0; i < dp.phases.size(); ++i) {
    DevPhase& a = dp.phases[i];
    const size_t last = std::min(dp.phases.size() - 1, i + a.merged);
    const bool head = i >= member_until;
    if (head) member_until = last + 1;
    a.fold_index = (head && a.fold_after_us > 0.f && a.fold_trigger != kTriggerEnd) ? dp.fold_count++ : -1;
    a.group_fold_us = 0.f;
    for (size_t q = i; q <= last; ++q) {
      const DevPhase& b = dp.phases[q];
      a.group_fold_us = std::max(a.group_fold_us, b.fold_after_us);
      if (b.resolves == static_cast<uint8_t>(Dependency::Router)) {
        a.group_router_phase = static_cast<int32_t>(q);
        for (int k = 0; k < b.n_pools; ++k)
          if (dp.pools[static_cast<size_t>(b.pools[k])].kind == kPoolRouter) a.group_router_pool = b.pools[k];
      }
    }
    if (a.group_router_phase >= 0) {
      const DevLayer& dl = dp.layers[static_cast<size_t>(a.layer)];
      for (int q = static_cast<int>(last) + 1; q < dl.phase_begin + dl.phase_count && a.group_routed_pool < 0; ++q)
        if (dp.phases[static_cast<size_t>(q)].routed >= 0) a.group_routed_pool = dp.phases[static_cast<size_t>(q)].pools[dp.phases[static_cast<size_t>(q)].routed];
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
