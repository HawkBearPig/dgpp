// The step engine's streamer (kernels/step/stream.hpp).
#include "kernels/step/stream.hpp"

#include "kernels/step/fp8_dense.cuh"
#include "kernels/step/mhc.cuh"
#include "kernels/csa2_attn_finish.cuh"
#include "kernels/csa2_norm.cuh"
#include "kernels/dsa.hpp"
#include "kernels/dsa_attn_flash.cuh"
#include "kernels/dsa_latent_append.cuh"
#include "net/bus_write.cuh"
#include "kernels/step/mxfp4_expert.cuh"
#include "kernels/step/mxfp4_row.cuh"
#include "kernels/step/router.cuh"

namespace dgpp::step {

namespace {

constexpr unsigned kStateIdle = 0, kStateStream = 1, kStateEnd = 2;
// The MXFP4 consumer's shape compiled into this kernel: DeepSeek-V4.1 at
// world 4. Other shapes come with their family ports.
constexpr int kConsumerH = 5120, kConsumerI = 576;
constexpr int kConsumerFp8Block = 32;  // DeepSeek-V4.1's fp8 scale grid
// Projections for groups of at most kFp8SplitRows slots take the split form:
// 32-row units, the width in four parts, when the width fits its staging.
// Measured 2026-10-30 against the 128-row slabbed form: the split form's
// drain is a third as long but its units cost more per byte (the staging
// of the whole slice a unit, the cross-warp reduce, four times the unit
// prologues), and the phases came out 20 percent longer, so it stays an
// option until a phase shape wants it.
constexpr bool kUseSplitForm = false;
constexpr int kSplitRowTiles = 2;
constexpr int kSplitUnitRows = kTileRows * kSplitRowTiles;
constexpr int kSplitMaxK = 5120;
__device__ __forceinline__ bool projection_split(const StepStreamParams& p, const DevPool& pool) {
  return kUseSplitForm && pool.kind == kPoolFp8Matrix && p.ctl->rows <= kFp8SplitRows && pool.k <= kSplitMaxK;
}

struct OutIndex {
  int hidden, spt;
  __device__ size_t operator()(int slot, int row) const { return step_out_index(slot, row, hidden, spt); }
};

__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ unsigned ld_acq(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_rel(unsigned* p, unsigned v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
// A unit's completion signal: a release add, after the block's barrier, publishes every thread's
// writes before it (a full fence and a plain add did the same at a higher cost under a saturated
// memory). The readers load with acquire.
__device__ __forceinline__ void red_release(unsigned* p, unsigned v) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned atom_add_acq_rel(unsigned* p, unsigned v) {
  unsigned old;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
  return old;
}
__device__ __forceinline__ unsigned ld_acq_sys(const volatile unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint64_t ld_acq_sys64(const volatile uint64_t* p) {
  uint64_t v;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_rel_sys64(volatile uint64_t* p, uint64_t v) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ unsigned word(int pass, int phase, unsigned state) {
  return (static_cast<unsigned>(pass) << 22) | (static_cast<unsigned>(phase) << 2) | state;
}
__device__ __forceinline__ void spin_us(float us) {
  if (us <= 0.f) return;
  const unsigned long long t0 = gtimer();
  const auto ns = static_cast<unsigned long long>(us * 1000.f);
  while (gtimer() - t0 < ns) {
  }
}

// The units of phase P of `layer` at the current run list.
// A pool's units this step: the run list's groups for the routed pool, one group otherwise.
// A pool's compute units a group this step: the plan's, or the split form's for a small group.
// The attention's (row, head) blocks of a step: 32 M-rows each (kernels/dsa_attn_flash.cuh dense::kM).
static_assert(dgpp::dsa_flash::dense::kM == 32 && dgpp::dsa_flash::dense::kThreads == kStepThreads, "the flash body's block is the step kernel's");
static_assert(kAttnGroup == kAttnSplitGroup && kAttnLatent == kCsa2Latent && kAttnRope == kCsa2Rope && kAttnIndexDim == kCsa2IndexDim,
              "the attention pools' geometry is the production kernels'");
__device__ __forceinline__ unsigned attn_blocks(const StepStreamParams& p) {
  return (p.ctl->rows * static_cast<unsigned>(p.attn.local_heads) + 31u) / 32u;
}
__device__ __forceinline__ unsigned compute_units_per_group(const StepStreamParams& p, const DevPool& pool) {
  if (projection_split(p, pool)) return (static_cast<unsigned>(pool.rows) + kSplitUnitRows - 1) / kSplitUnitRows;
  if (pool.kind == kPoolMhc) return static_cast<unsigned>(pool.units_per_group) * p.ctl->rows;  // a token's units for every token
  if (pool.kind == kPoolRowTail) return p.ctl->rows;  // a row a unit
  if (pool.kind == kPoolAttention) return static_cast<unsigned>(pool.rows) * attn_blocks(p);  // a split of every (row, head) block
  return static_cast<unsigned>(pool.units_per_group);
}
// A slabbed projection's reduce units: handed out after every pool's compute
// units and the fixed units (below), so the pool they wait on has only its
// stragglers left; handed out right after their own pool's compute units they
// spun on it, 58 of qkv's 211 units on DeepSeek-V4.1 (2026-10-30). A single
// worker reducing as the pool's last arriver was worse: the largest pool's
// reduce became the phase's tail.
__device__ __forceinline__ unsigned reduce_units(const StepStreamParams& p, const DevPool& pool) {
  if (pool.kind == kPoolAttention) return pool.gate_arg ? p.ctl->rows * static_cast<unsigned>(p.attn.local_heads) : 0u;  // the finish: one (row, head) a unit
  if (pool.kind != kPoolFp8Matrix || pool.slabs <= 1 || projection_split(p, pool) || pool.reduce_by_tail) return 0;
  return (static_cast<unsigned>(pool.rows) * static_cast<unsigned>(p.ctl->rows) + kStepAccChunk - 1) / kStepAccChunk;
}
__device__ __forceinline__ unsigned pool_units(const StepStreamParams& p, int layer, const DevPool& pool) {
  const unsigned groups = pool.group_kind == kGroupRouted ? static_cast<unsigned>(p.ctl->run_count[layer]) : 1u;
  return groups * compute_units_per_group(p, pool) + reduce_units(p, pool);
}
// The ordered accumulation's chunks (the accumulate phase's own units, and its fold's trigger).
__device__ __forceinline__ unsigned accumulate_units(const StepStreamParams& p) {
  return (static_cast<unsigned>(p.ctl->rows) * static_cast<unsigned>(p.hidden) + kStepAccChunk - 1) / kStepAccChunk;
}
// The group's fold on the bus (world > 1): its source rows, the partial the update-carrying
// projection's reduce units (attn_out: its y) or the accumulation (its y) complete, [rows][N]
// bf16 contiguous; and its unit count, the write transport's grid for that many elements.
__device__ __forceinline__ bool fold_source(const StepStreamParams& p, int P, uint16_t** y, unsigned* elems) {
  const DevPhase& ph = p.phases[P];
  if (ph.accumulate) {
    *y = p.y;
    *elems = p.ctl->rows * static_cast<unsigned>(p.hidden);
    return p.y != nullptr;
  }
  for (int i = 0; i < ph.n_pools; ++i) {
    const DevPool& pool = p.pools[ph.pools[i]];
    if (pool.kind == kPoolFp8Matrix && pool.update_site) {
      *y = p.proj + pool.out_offset;
      *elems = p.ctl->rows * static_cast<unsigned>(pool.rows);
      return p.proj != nullptr;
    }
  }
  return false;
}
__device__ __forceinline__ unsigned fold_units(const StepStreamParams& p, int P) {
  const DevPhase& ph = p.phases[P];
  if (p.fold.world <= 1 || ph.fold_index < 0 || ph.fold_trigger == kTriggerEnd) return 0u;
  uint16_t* y = nullptr;
  unsigned elems = 0;
  if (!fold_source(p, P, &y, &elems)) return 0u;
  return net::bus_write_blocks((elems + net::kBusWriteChunkElems - 1) / net::kBusWriteChunkElems);  // bus_write_chunks is host-side
}
__device__ __forceinline__ unsigned own_units(const StepStreamParams& p, int layer, int P) {
  const DevPhase& ph = p.phases[P];
  const unsigned fold = 2u * fold_units(p, P);  // the stage class and the fold class of the group's fold
  if (ph.accumulate) return accumulate_units(p) + fold;
  unsigned n = ph.fixed_units + fold;
  for (int i = 0; i < ph.n_pools; ++i) {
    const DevPool& pool = p.pools[ph.pools[i]];
    n += pool_units(p, layer, pool);
    if (pool.gate_kind == kGatePool) n += pool.stage_units;  // its staging class (closed early, its rest is credited as done)
  }
  return n;
}
__device__ __forceinline__ unsigned pool_compute_units(const StepStreamParams& p, int layer, const DevPool& pool) {
  const unsigned groups = pool.group_kind == kGroupRouted ? static_cast<unsigned>(p.ctl->run_count[layer]) : 1u;
  return groups * compute_units_per_group(p, pool);
}
// A pool's y complete this step: its compute units, and its reduce units when it is slabbed (they
// sum the slabs into y). The gate of the classes that read it (kGatePool).
__device__ __forceinline__ bool pool_complete(const StepStreamParams& p, int layer, int pool_id) {
  const DevPool& q = p.pools[pool_id];
  if (q.kind == kPoolFp8Matrix && q.slabs > 1 && !projection_split(p, q) && !q.reduce_by_tail)
    return ld_acq(&p.ctl->reduce_done[layer][pool_id]) >= reduce_units(p, q);
  return ld_acq(&p.ctl->pool_done[layer][pool_id]) >= pool_compute_units(p, layer, q);
}
// A kGatePool class's gate: the pool whose y it reads is complete, and its other gate.
__device__ __forceinline__ bool pool_gate_open(const StepStreamParams& p, int layer, const DevPool& q) {
  return pool_complete(p, layer, q.in_pool) && (q.gate_pool < 0 || pool_complete(p, layer, q.gate_pool));
}
// A block-diagonal projection's head group of unit rb (device_plan.hpp in_group_rows): the
// group whose columns the unit's rows read, and the key of its staged activations.
__device__ __forceinline__ int projection_diag_group(const StepStreamParams& p, const DevPool& pool, int rb) {
  const bool split = projection_split(p, pool);
  const int block = split ? rb : rb / pool.slabs;
  return (block * (split ? kSplitUnitRows : pool.unit_rows)) / pool.in_group_rows;
}
// The published group's unit classes, built once by the chain into the control block before the
// group is published and read by each worker in one round trip (built by every worker from the
// pool tables it cost each up to 17 µs a phase under a saturated memory, 2026-10-31). An entry:
// x the unit count, y grab | kind << 8 | gate_kind << 16 | gate_arg << 24, z phase | pool << 16
// (pool as a signed 16-bit field), in phase order: a pool's compute units, that pool's reduce
// units, the fixed byte units; an accumulation's chunks. (Gated classes first, so the merged sites
// and router ran as soon as they opened, measured 3.6 points worse on 2026-10-31: under the
// saturated stream their latency-bound units ran slower than in the drain, and every grab paid
// their gate loads. Merged phases first deadlocked: the down units, waited per expert group
// inside the unit, held every worker ahead of the gate and up units.)
constexpr uint8_t kClassCompute = 0, kClassReduce = 1, kClassFixed = 2, kClassAccumulate = 3, kClassStage = 4, kClassFoldStage = 6, kClassFold = 7;
constexpr uint8_t kGateOwnPool = 3;  // a reduce class: its pool's compute units complete
constexpr uint8_t kGateTrigger = 4;  // a staging class: the group's fold trigger holds (the fold's idle window begins)
constexpr uint8_t kGateAttnPools = 5;  // the attention finish: every attention pool of the phase has its blocks complete
constexpr uint8_t kGateFoldStaged = 6;  // the fold units: every stage unit of the group's fold published its row
__device__ void build_class_table(const StepStreamParams& p, int P) {
  StepCtl* ctl = p.ctl;
  const unsigned rows = ctl->rows;
  unsigned n = 0;
  // The gated pools' staging classes, gathered while the pools are walked once (a second walk of
  // the pool records cost the chain about 9 µs a group under the stream, 2026-11-01) and emitted
  // behind the phase's own classes.
  __shared__ int sm_stage_pool[kMaxPhasePools];
  __shared__ unsigned sm_stage_units[kMaxPhasePools], sm_stage_base[kMaxPhasePools];
  int n_stage = 0;
  // w = 1: the count is a group's units, the worker multiplies by the layer's run count when it
  // loads the table, so the table is built once before the select and never waits on it.
  auto put = [&](unsigned count, unsigned grab, uint8_t kind, int phase, int pool, uint8_t gate_kind, uint8_t gate_arg, unsigned per_run) {
    if (n >= static_cast<unsigned>(kStepMaxClasses)) return;
    uint4 e;
    e.x = count;
    e.y = grab | (static_cast<unsigned>(kind) << 8) | (static_cast<unsigned>(gate_kind) << 16) | (static_cast<unsigned>(gate_arg) << 24);
    e.z = static_cast<unsigned>(phase) | (static_cast<unsigned>(pool & 0xffff) << 16);
    e.w = per_run;
    ctl->cls_table[P][n++] = e;
  };
  // Phase order (gated classes first lost 3.6 points, merged phases first deadlocked:
  // docs/step_engine.md); within a merged phase its own classes before its staging class.
  for (int ip = P; ip <= P + static_cast<int>(p.phases[P].merged); ++ip) {
    const DevPhase& iph = p.phases[ip];
    if (iph.accumulate) {  // the ordered accumulation's chunks are the phase's own units
      put(accumulate_units(p), 1u, kClassAccumulate, ip, -1, kGateNone, 0, 0u);
    } else {
      for (int rp = 0; rp < 2; ++rp)
        for (int i = 0; i < iph.n_pools; ++i) {
          if ((rp == 0) == (i == iph.routed)) continue;  // the routed pool after the others
          const DevPool& gp = p.pools[iph.pools[i]];
          const bool routed = gp.group_kind == kGroupRouted;
          const unsigned per_group = compute_units_per_group(p, gp);
          if (per_group)
            put(per_group, gp.kind == kPoolBytes ? 1u : (projection_split(p, gp) ? 4u : static_cast<unsigned>(gp.grab)), kClassCompute, ip, iph.pools[i],
                gp.gate_kind, gp.gate_arg, routed ? 1u : 0u);
          if (gp.gate_kind == kGatePool && gp.stage_units && n_stage < kMaxPhasePools) {
            sm_stage_pool[n_stage] = iph.pools[i];
            sm_stage_units[n_stage] = gp.stage_units;
            sm_stage_base[n_stage] = gp.stage_unit_begin;
            ++n_stage;
          }
          // Its reduce units, gated on its compute units: the update-carrying projection's y (and
          // the attention update) completes early in the group.
          const unsigned reduce = reduce_units(p, gp);
          if (reduce) put(reduce, 1u, kClassReduce, ip, iph.pools[i], gp.kind == kPoolAttention ? kGateAttnPools : kGateOwnPool, 0, 0u);
        }
      if (iph.fixed_units) put(iph.fixed_units, 1u, kClassFixed, ip, -1, kGateNone, 0, 0u);  // the shortest units, they fill the tail
      // A gated pool's bytes staged into L2 while its gate is closed (the chain's dependent
      // projections: wq_b behind the q norm, wo_b behind wo_a): the workers the producer's tail
      // leaves idle stream them with the evict-last policy, so the pool's units read hot lines when
      // the gate opens; the class closes when it does (its rest credited as done). Read by the
      // pool's own units alone, wq_b's 10.5 MB took 46-57 µs behind the norm (2026-11-01).
      for (int i = 0; i < n_stage; ++i) put(sm_stage_units[i], 1u, kClassStage, ip, sm_stage_pool[i], kGateNone, 0, sm_stage_base[i] << 8);
      n_stage = 0;
    }
    // The group's fold on the bus (world > 1): the stage units open on the trigger (the partial
    // complete) and lay it out in the generation's staging row; the fold units open when every
    // stage unit has published, poll the peers' rows, fold in rank order and apply the update;
    // the last sets fold_done, the gate of the merged classes behind the fold.
    if (ip == P) {
      const unsigned fu = fold_units(p, P);
      if (fu) {
        put(fu, 1u, kClassFoldStage, ip, -1, kGateTrigger, 0, 0u);
        put(fu, 1u, kClassFold, ip, -1, kGateFoldStaged, 0, 0u);
      }
    }
    // A merged phase's lead-only bytes (a site's coefficient rows, the router's rows) are staged
    // into L2 by a class of the group that opens on the fold's trigger, so they are read during
    // the fold's idle window and are hot when the phase's own classes open behind the fold.
    // Staged by the lead early in the group they were gone by then: about 30 MB of stream and
    // lead traffic pass through the 24 MB L2 in between, and the router's units ran at three
    // times their standalone time (2026-11-01). The class follows the phase's own: once a gated
    // class is open its units come first, and the staging takes only the workers they leave
    // (ahead of them, the router's 61 staging units of 64 KB were still being handed out when
    // its gate opened, and its own units waited 10 µs for workers).
    if (ip > P && iph.lead_units > iph.fixed_units) {
      // Gated on the fold's trigger when the group has one; a group without (the expert pair)
      // stages at once, while its first phase streams.
      const uint8_t gate = p.phases[P].fold_trigger != kTriggerEnd ? kGateTrigger : kGateNone;
      put(iph.lead_units - iph.fixed_units, 1u, kClassStage, ip, -1, gate, 0, static_cast<unsigned>(iph.fixed_units) << 8);
    }
  }
  (void)rows;
  ctl->cls_n[P] = n;
}
// The units a published phase hands out: its own and its merged phases', every class.
__device__ __forceinline__ unsigned phase_units(const StepStreamParams& p, int P) {
  unsigned n = 0;
  for (int Q = P; Q <= P + static_cast<int>(p.phases[P].merged); ++Q) {
    n += own_units(p, p.phases[Q].layer, Q);
    if (Q > P) n += p.phases[Q].lead_units - p.phases[Q].fixed_units;  // the merged phase's staging class (build_class_table)
  }
  return n;
}
// Whether the group's fold may be spun: its trigger holds.
__device__ __forceinline__ bool fold_trigger_holds(const StepStreamParams& p, int layer, int P) {
  const DevPhase& ph = p.phases[P];
  if (ph.fold_trigger == kTriggerReduce && ph.trigger_pool >= 0)
    return ld_acq(&p.ctl->reduce_done[layer][ph.trigger_pool]) >= reduce_units(p, p.pools[ph.trigger_pool]);
  if (ph.fold_trigger == kTriggerAccumulate) return ld_acq(&p.ctl->acc_done[layer]) >= accumulate_units(p);
  return false;
}

__device__ __forceinline__ bool locate_segment(const StepStreamParams& p, const uint8_t* base, const DevSegment& s, unsigned u,
                                               const uint8_t** ptr, unsigned* bytes) {
  if (u < s.unit_begin || u >= s.unit_begin + s.units) return false;
  const unsigned long long off = s.offset + static_cast<unsigned long long>(u - s.unit_begin) * p.unit_bytes;
  const unsigned long long left = s.offset + s.bytes - off;
  *ptr = base + off;
  *bytes = static_cast<unsigned>(left < p.unit_bytes ? left : p.unit_bytes);
  return true;
}
// Lead unit u of phase P: the fixed segments, then the lead-only ones.
// Lead unit u of phase P: the fixed segments, then the lead-only ones, in
// the consumers' order. (The lead walking a phase's even units first and
// then its odd ones, so a partly staged phase ran hot and cold units side by
// side, measured 1.3 points worse over three clean runs on 2026-10-31.)
__device__ bool locate_lead(const StepStreamParams& p, int layer, int P, unsigned u, const uint8_t** ptr, unsigned* bytes) {
  const DevPhase& ph = p.phases[P];
  const uint8_t* base = p.bases[layer];
  for (int i = 0; i < ph.seg_count; ++i)
    if (locate_segment(p, base, p.segments[ph.seg_begin + i], u, ptr, bytes)) return true;
  for (int i = 0; i < ph.lead_seg_count; ++i)
    if (locate_segment(p, base, p.segments[ph.lead_seg_begin + i], u, ptr, bytes)) return true;
  return false;
}
// Fixed byte unit v of phase P in `layer`: its bytes in the layer's span.
__device__ bool locate_fixed(const StepStreamParams& p, int layer, int P, unsigned v, const uint8_t** ptr, unsigned* bytes) {
  const DevPhase& ph = p.phases[P];
  const uint8_t* base = p.bases[layer];
  if (v >= ph.fixed_units) return false;
  for (int i = 0; i < ph.seg_count; ++i)
    if (locate_segment(p, base, p.segments[ph.seg_begin + i], v, ptr, bytes)) return true;
  return false;
}
// Byte unit v of a byte pool (the synthetic routed pool): group v / units_per_group of the run list.
__device__ bool locate_pool_bytes(const StepStreamParams& p, int layer, const DevPool& pool, unsigned v, const uint8_t** ptr, unsigned* bytes) {
  const uint8_t* base = p.bases[layer];
  const int g = static_cast<int>(v / static_cast<unsigned>(pool.units_per_group));
  const unsigned t = v % static_cast<unsigned>(pool.units_per_group);
  if (g >= p.ctl->run_count[layer]) return false;
  const int group = p.ctl->run_list[layer][g];
  const unsigned long long inner = static_cast<unsigned long long>(t) * p.unit_bytes;
  const unsigned long long left = pool.bytes - inner;
  *ptr = base + pool.base + static_cast<unsigned long long>(group) * pool.stride + inner;
  *bytes = static_cast<unsigned>(left < p.unit_bytes ? left : p.unit_bytes);
  return true;
}

// The phase that holds global lead unit `g` (the phases' lead_begin ascend).
// The phase a lead unit belongs to, over a shared-memory copy of the phases' lead_begin: the workers'
// lead hand-out ran two of these searches over global memory a unit, about
// fourteen dependent loads, which held the lead to half the ceiling in a
// ten-microsecond latency window (2026-10-31).
__device__ __forceinline__ int phase_of_lead_unit_sm(const unsigned* lead_begin, int n_phases, unsigned g) {
  int lo = 0, hi = n_phases - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (lead_begin[mid] <= g) lo = mid; else hi = mid - 1;
  }
  return lo;
}

// Streams `bytes` from `ptr` through the block: lane-contiguous 16-byte
// loads, four in flight a thread, folded into one word so the loads live.
constexpr unsigned kLeadPiece = 16u << 10;  // a lead unit streams in pieces of this size and may stop between them
// The lead's load: into L2 with the evict-last priority, so the staged line
// outlives the evict-first stream and the normal traffic until its consumer
// reads it with an evict-first load, which demotes it.
__device__ __forceinline__ uint4 ld_stage(const uint4* p) {
  uint4 v;
  unsigned long long pol;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  asm volatile("ld.global.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p), "l"(pol));
  return v;
}
template <bool kEvictFirst>
__device__ unsigned stream_unit(const uint8_t* ptr, unsigned bytes) {
  const uint4* q = reinterpret_cast<const uint4*>(ptr);
  const unsigned n = bytes / 16;
  unsigned acc = 0;
  unsigned i = threadIdx.x;
  for (; i + 3 * kStepThreads < n; i += 4 * kStepThreads) {
    uint4 a, b, c, d;
    if (kEvictFirst) {
      a = __ldcs(q + i); b = __ldcs(q + i + kStepThreads); c = __ldcs(q + i + 2 * kStepThreads); d = __ldcs(q + i + 3 * kStepThreads);
    } else {
      a = ld_stage(q + i); b = ld_stage(q + i + kStepThreads); c = ld_stage(q + i + 2 * kStepThreads); d = ld_stage(q + i + 3 * kStepThreads);
    }
    acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.y ^ c.z ^ c.w ^ d.x ^ d.y ^ d.z ^ d.w;
  }
  for (; i < n; i += kStepThreads) {
    const uint4 a = kEvictFirst ? __ldcs(q + i) : ld_stage(q + i);
    acc ^= a.x ^ a.y ^ a.z ^ a.w;
  }
  return acc;
}

// The synthetic router: `rows` rows pick `top_k` distinct groups each (a
// counter-based hash per pass, layer and row), the distinct picks sorted
// ascending into the layer's run list. Block-wide, the chain block's. With
// `route_pool` set, every row of a layer picks from the same window of that
// many experts, which gives the rows the overlap real routing has: a window
// of 32 yields 19.7 distinct experts for five rows and 10.1 for two, as
// measured on DeepSeek-V4.1 (2026-10-30), where rows picking from all 384
// yield 28.9 and 11.8.
__device__ void route_synthetic_picks(const StepStreamParams& p, int layer, const DevPool& pool, int* keys) {
  const int rows = static_cast<int>(p.ctl->rows), top_k = static_cast<int>(p.ctl->top_k);
  if (threadIdx.x < static_cast<unsigned>(rows)) {
    for (int i = 0; i < top_k; ++i) p.ctl->slot_weight[layer][threadIdx.x * (top_k + 1) + i] = 1.0f / static_cast<float>(top_k);
    p.ctl->slot_weight[layer][threadIdx.x * (top_k + 1) + top_k] = 1.0f;
    unsigned h = p.ctl->seed * 2654435761u + 7919u * static_cast<unsigned>(layer) + 104729u * threadIdx.x;
    const unsigned window = (p.ctl->route_pool > 0 && p.ctl->route_pool < static_cast<unsigned>(pool.groups)) ? p.ctl->route_pool
                                                                                                             : static_cast<unsigned>(pool.groups);
    const unsigned start = window < static_cast<unsigned>(pool.groups)
                               ? ((p.ctl->seed * 40503u + 7919u * static_cast<unsigned>(layer)) % (static_cast<unsigned>(pool.groups) - window + 1u))
                               : 0u;
    for (int i = 0; i < top_k; ++i) {
      int e;
      bool dup;
      do {
        h = h * 1664525u + 1013904223u;
        e = static_cast<int>(start + (h >> 8) % window);
        dup = false;
        for (int j = 0; j < i; ++j) dup = dup || keys[threadIdx.x * top_k + j] == e;
      } while (dup);
      keys[threadIdx.x * top_k + i] = e;
    }
  }
  __syncthreads();
}
// From keys [rows][top_k] (a row's picks, any order): the run list of distinct
// experts ascending, each expert's slots and tokens, the shared slots.
// `smem`: the chain's dynamic shared memory past the select's rows (static shared memory is at
// the device's 100 KB block limit with the consumers' dynamic region: a static array here failed
// the launch, 2026-10-31).
__device__ void route_finish(const StepStreamParams& p, int layer, int* keys, int* sorted, uint8_t* smem) {
  int* s_run = reinterpret_cast<int*>(smem);  // the run list in shared memory: no global round trip to read it back
  int& s_n = s_run[kStepMaxRows * 8];
  const int rows = static_cast<int>(p.ctl->rows), top_k = static_cast<int>(p.ctl->top_k);
  const int picks = rows * top_k;
  if (threadIdx.x < static_cast<unsigned>(picks)) {
    const int k = keys[threadIdx.x];
    int pos = 0;
    for (int o = 0; o < picks; ++o) pos += (keys[o] < k) || (keys[o] == k && o < static_cast<int>(threadIdx.x));
    sorted[pos] = k;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int n = 0, prev = -1;
    for (int o = 0; o < picks; ++o)
      if (sorted[o] != prev) { s_run[n] = sorted[o]; p.ctl->run_list[layer][n] = sorted[o]; ++n; prev = sorted[o]; }
    s_n = n;
    p.ctl->run_count[layer] = n;
  }
  __syncthreads();
  // Group g: the slots of run_list[g], slot = row * (top_k + 1) + pick, in row order.
  if (threadIdx.x < static_cast<unsigned>(s_n)) {
    const int g = static_cast<int>(threadIdx.x), expert = s_run[g];
    int n = 0;
    for (int o = 0; o < picks && n < 32; ++o)
      if (keys[o] == expert) {
        p.ctl->group_slots[layer][g][n] = (o / top_k) * (top_k + 1) + (o % top_k);
        p.ctl->group_tokens[layer][g][n] = o / top_k;
        ++n;
      }
    p.ctl->group_count[layer][g] = n;
    p.ctl->gate_done[layer][g] = 0;
  }
  if (threadIdx.x < static_cast<unsigned>(rows)) {
    p.ctl->shared_slots[layer][threadIdx.x] = static_cast<int>(threadIdx.x) * (top_k + 1) + top_k;
    p.ctl->shared_tokens[layer][threadIdx.x] = static_cast<int>(threadIdx.x);
    p.ctl->token_slots[threadIdx.x] = static_cast<int>(threadIdx.x);
  }
  if (threadIdx.x == 0) p.ctl->shared_gate_done[layer] = 0;
  __syncthreads();
}

// The ordered accumulation of one chunk of outputs: the routed slots of a
// token ascending with weight 1 / top_k each, then the shared slot with
// weight 1, as the slot kernels' accumulation does (synthetic weights until
// the router's are real).
// With `fused`, the MoE site's stream update follows on the same chunk: the
// layer's output y is the update's h, read back as the bf16 the production
// path rounds it to, streams[1] (the site's input) mix into streams[0].
// An output's slots are contiguous in the partials (step_out_index), so a
// thread issues every slot's load before its first fma, and four outputs'
// loads together; a slot-by-slot loop ran 15 µs a unit at one block an SM.
constexpr int kStepMaxSlotsPerToken = 9;
__device__ void accumulate_chunk(const StepStreamParams& p, int layer, unsigned unit, bool fused) {
  const int rows = static_cast<int>(p.ctl->rows), top_k = static_cast<int>(p.ctl->top_k), spt = top_k + 1, H = p.hidden;
  const int i0 = static_cast<int>(unit) * kStepAccChunk, i1 = min(i0 + kStepAccChunk, rows * H);
  const float* __restrict__ sw = p.ctl->slot_weight[layer];
  const float* __restrict__ out = p.out;
  uint16_t* __restrict__ y = p.y;
  // Four outputs a round: every slot of the four loaded, then the fmas (y and the partials
  // may alias as far as the compiler knows, so the pipelining is written out).
  constexpr int kRound = 4;
  for (int i = i0 + static_cast<int>(threadIdx.x); i < i1; i += kRound * kStepThreads) {
    float v[kRound][kStepMaxSlotsPerToken];
#pragma unroll
    for (int r = 0; r < kRound; ++r) {
      const int ir = i + r * kStepThreads;
      const float* src = out + static_cast<size_t>(ir) * spt;  // slot j of output (t, c): (t * H + c) * spt + j
#pragma unroll
      for (int j = 0; j < kStepMaxSlotsPerToken; ++j) v[r][j] = (ir < i1 && j < spt) ? src[j] : 0.f;
    }
#pragma unroll
    for (int r = 0; r < kRound; ++r) {
      const int ir = i + r * kStepThreads;
      if (ir >= i1) break;
      const int t = ir / H;
      float a = 0.f;
#pragma unroll
      for (int j = 0; j < kStepMaxSlotsPerToken; ++j)
        if (j < spt) a = __fmaf_rn(sw[t * spt + j], v[r][j], a);
      y[ir] = f32_to_bf16(a);
    }
  }
  if (fused) mhc_update_chunk(p.post, p.comb, p.y, p.streams[1], p.streams[0], H, i0, i1);  // the same thread owns the same elements (the caller withholds it when the group's fold units carry it)
}

// A hyper-connection site's unit: kMhcUnitCoeffs logits of one token (unit
// rb = token * units + group), the arrival that completes the token's 24
// runs its finish: pre, post, the Sinkhorn comb and the normed row into x
// (each unit collapsed its slice of the row on the way). Site s reads
// streams[s] and pre[s], writes pre[s ^ 1]. Two barriers to the ticket and
// three more in the finishing block (2026-11-01; twenty before: the norm's,
// three dots' and the finish's reductions each had their own, the collapse
// its own loads, the finish read the row twice and ran the Sinkhorn before
// the norm). `tr`: the unit's stage stamps (kMhcTraceWords), when traced.
__device__ void mhc_unit(const StepStreamParams& p, int layer, const DevPool& pool, const uint8_t* base, int rb, uint8_t* smem,
                         unsigned long long* tr) {
  __shared__ unsigned sm_last;
  constexpr int kUnits = kMhcCoeffs / kMhcUnitCoeffs;  // a token's units
  const int H = p.hidden, s = pool.site;
  const int t = rb / kUnits, u = rb % kUnits, c0 = u * kMhcUnitCoeffs;
  if (tr && threadIdx.x == 0) tr[0] = gtimer();
  const uint16_t* streams = p.streams[s] + static_cast<size_t>(t) * kMhcN * H;
  const uint16_t* fn = reinterpret_cast<const uint16_t*>(base + pool.offsets[0]) + static_cast<size_t>(c0) * kMhcN * H;
  MhcUnitSmem<kMhcUnitCoeffs>* sm = reinterpret_cast<MhcUnitSmem<kMhcUnitCoeffs>*>(smem);
  float* f = reinterpret_cast<float*>(smem + kMhcUnitSmem);  // the finish's: [0, 24) the token's logits, [24, 40) the comb seed, [40, 48) scratch
  float lg[kMhcUnitCoeffs];
  uint16_t* row = p.x + static_cast<size_t>(t) * p.x_stride;
  const int vecs = H / 8, per = (vecs + kUnits - 1) / kUnits;
  mhc_dot_n<kMhcUnitCoeffs>(streams, fn, H, p.mhc_norm_eps, sm, lg, row, p.pre[s] + t * kMhcN, u * per, min(vecs, (u + 1) * per));
  // Every thread passed the dots' barrier after its slice stores: the ticket's release covers them.
  if (threadIdx.x == 0) {
    if (tr) tr[1] = gtimer();
#pragma unroll
    for (int c = 0; c < kMhcUnitCoeffs; ++c) p.logits[t * kMhcCoeffs + c0 + c] = lg[c];
    const unsigned before = atom_add_acq_rel(&p.ctl->mhc_done[layer][s][t], static_cast<unsigned>(kMhcUnitCoeffs));
    const bool last = before + kMhcUnitCoeffs == static_cast<unsigned>(kMhcCoeffs);
    sm_last = last ? 1u : 0u;
    if (last) p.ctl->mhc_done[layer][s][t] = 0;  // every logit of the token arrived: ready for the next step
    if (tr) tr[2] = gtimer();
  }
  __syncthreads();
  if (!sm_last) return;
  MhcSite site;
  site.base = reinterpret_cast<const float*>(base + pool.offsets[1]);
  site.scale = reinterpret_cast<const float*>(base + pool.offsets[2]);
  site.ln = reinterpret_cast<const uint16_t*>(base + pool.offsets[3]);
  site.hc_eps = p.hc_eps;
  site.sinkhorn_iters = p.sinkhorn_iters;
  site.ln_eps = p.mhc_norm_eps;
  // The normed row first (one round trip past L1 for the logits, the row and the norm weight
  // together), released to the router as soon as it is stored; the token's coefficients for the
  // later update follow on warp 0, off the router's path (the Sinkhorn is about 6 µs).
  mhc_finish_norm(row, p.logits + t * kMhcCoeffs, f, site, H, f + kMhcCoeffs + kMhcN * kMhcN, tr);
  __syncthreads();  // the row is stored by every thread; the logits are in shared memory
  if (threadIdx.x == 0) {
    red_release(&p.ctl->site_done[layer][s], 1u);  // the router's gate
    red_release(&p.ctl->open_gen, 1u);             // lead units yield
    if (tr) tr[6] = gtimer();
  }
  mhc_finish_coefficients(f, site, p.pre[s ^ 1] + t * kMhcN, p.post + t * kMhcN, p.comb + t * kMhcN * kMhcN, f + kMhcCoeffs, tr);
  __syncthreads();  // the coefficients are stored before the unit counts as done
  if (tr && threadIdx.x == 0) tr[kMhcTraceWords - 1] = gtimer();
}

// Whether every attention pool of a phase has its blocks complete (the finish's gate).
__device__ __forceinline__ bool attn_pools_done(const StepStreamParams& p, int layer, int phase) {
  const DevPhase& ph = p.phases[phase];
  for (int i = 0; i < ph.n_pools; ++i) {
    const DevPool& q = p.pools[ph.pools[i]];
    if (q.kind != kPoolAttention) continue;
    if (ld_acq(&p.ctl->pool_done[layer][ph.pools[i]]) < pool_compute_units(p, layer, q)) return false;
  }
  return true;
}

// The CSA2 attention's unit: one block of the production flash kernel over
// the pool's latent source (kernels/dsa_attn_flash.cuh attn_flash_body): the
// (row, head) block bx = unit % blocks, the split s = unit / blocks (a block's
// units in sequence share its queries in L2). The window ring's rows are the
// fp8_block form behind the identity table (request r's region r, bounded by
// the request count); the main cache's the fp4_block form behind the paged
// block table. The group merge's tickets and partials are the step's. Bitwise
// the production kernel's partials: the same body at the same coordinates.
// Out of line (2026-11-01): the attention and fold bodies run on a few units a layer, or not at
// all at world 1, and inlined they took the kernel's stack from 48 to 176 bytes and the rate band
// down about a point on the paths that never reach them; the expert units' hot loops stay inlined
// (out of line they lost 2.5 points, consumer_unit above).
__device__ __forceinline__ void attention_unit(const StepStreamParams& p, int layer, const DevPool& pool, int rb, uint8_t* smem, const uint16_t* q) {
  const StepAttnParams& a = p.attn;
  const int rows = static_cast<int>(p.ctl->rows);
  const int blocks = static_cast<int>(attn_blocks(p));
  const int bx = rb % blocks, s = rb / blocks;
  const int n_split = pool.rows, src = pool.site;
  const uint8_t* latent = nullptr;
  const int32_t* tables = nullptr;
  const int32_t* list = nullptr;
  const int32_t* counts = nullptr;
  int block_tokens = 0, bpr = 0, total = 0, stride = 0;
  if (src == 0) {
    latent = a.ring + static_cast<unsigned long long>(layer) * a.ring_layer_bytes;
    block_tokens = a.ring_slots; tables = a.ring_table; bpr = 1; total = a.max_requests;
    list = a.wlist; stride = a.window; counts = a.wcounts;
  } else {
    latent = a.main ? a.main[layer] : nullptr;
    block_tokens = a.epb ? a.epb[layer] : 0; tables = a.block_tables; bpr = a.blocks_per_request; total = a.blocks_total;
    list = a.topk; stride = a.index_topk; counts = a.counts;
  }
  if (latent == nullptr || q == nullptr || list == nullptr || counts == nullptr || block_tokens <= 0) {
    if (threadIdx.x == 0) atomicCAS(&p.ctl->status, kStepOk, kStepBadUnit);
    return;
  }
  const bool grouped = pool.slabs == kAttnSplitGroup;
#define DGPP_STEP_ATTN_ARGS                                                                                                   \
  bx, s, smem, q, latent, nullptr, a.req_ids, a.pos, list, stride, counts, rows, n_split, a.local_heads, block_tokens, tables, \
      bpr, total, a.scale, a.m[src], a.l[src], a.c[src], a.tickets[src], a.mg[src], a.lg[src], a.cg[src]
  // The step's form of the body (kernels/dsa_attn_flash.cuh): a tile's rows fetched in one pass,
  // four fold outputs a thread, the ring's identity table resolved without the table load; the
  // arithmetic the production's. At one block a worker under the lead's stream each dependent
  // round trip to L2 is about 2 µs, and the production's form had about sixteen (2026-11-01).
  using dgpp::dsa_flash::attn_flash_body;
  constexpr int kPasses = 1, kFoldPer = 4;
  if (src == 0) {
    if (grouped) attn_flash_body<kAttnLatent, kAttnLatent, true, LatentFormat::kFp8Block, kAttnSplitGroup, kPasses, kFoldPer, true>(DGPP_STEP_ATTN_ARGS);
    else attn_flash_body<kAttnLatent, kAttnLatent, true, LatentFormat::kFp8Block, 1, kPasses, kFoldPer, true>(DGPP_STEP_ATTN_ARGS);
  } else {
    if (grouped) attn_flash_body<kAttnLatent, kAttnLatent, true, LatentFormat::kFp4Block, kAttnSplitGroup, kPasses, kFoldPer, false>(DGPP_STEP_ATTN_ARGS);
    else attn_flash_body<kAttnLatent, kAttnLatent, true, LatentFormat::kFp4Block, 1, kPasses, kFoldPer, false>(DGPP_STEP_ATTN_ARGS);
  }
#undef DGPP_STEP_ATTN_ARGS
}

// The group's fold on the bus (world > 1; net/bus_write.cuh). The stage unit: block `unit` of the
// write transport's grid lays its chunks of the group's partial (the update-carrying projection's
// y, or the accumulation's) out in the generation's staging row with their trailers and publishes
// its record's ready; the host's engine posts the row to every peer when every block has. The
// generation comes from the fold table the host filled for this step (its slot) at the group's
// fold index. Bitwise the production fold: the same body as the stream collective's kernel.
__device__ __forceinline__ StepFoldDev fold_handle(const StepStreamParams& p, int P) {
  const volatile StepFoldDev* f = p.fold.table + static_cast<size_t>(p.ctl->fold_slot) * kStepMaxFolds + p.phases[P].fold_index;
  StepFoldDev h;
  h.gen = f->gen;
  h.elems = f->elems;
  h.cell = f->cell;
  h.recs = f->recs;
  return h;
}
__device__ __noinline__ void fold_stage_unit(const StepStreamParams& p, int P, unsigned unit) {
  uint16_t* y = nullptr;
  unsigned elems = 0;
  const unsigned n = fold_units(p, P);
  const StepFoldDev f = fold_handle(p, P);
  if (!fold_source(p, P, &y, &elems) || n == 0 || f.gen == 0 || f.recs == nullptr || f.elems != elems) {
    if (threadIdx.x == 0) atomicCAS(&p.ctl->status, kStepOk, kStepBadUnit);
    return;
  }
  net::bus_write::stage_block(static_cast<int>(unit), static_cast<int>(n), p.fold.view, f.gen, y, elems, f.recs[unit], p.ctl->fold_epoch);
  if (threadIdx.x == 0) red_release(&p.ctl->fold_staged[P], 1u);  // after stage_block's barrier and release of ready
}
// The fold unit: its chunks of every peer's row polled, proven and folded in rank order into the
// group's partial in place, then the residual update those elements' reduce or accumulate units
// carried at world 1, on the folded values (the attention site: streams[0] into streams[1] under
// the site's post and comb with h the projection's y; the MoE site: streams[1] into streams[0]
// with the accumulation's y). The last fold unit of the group sets fold_done and the group's
// stamps. A unit that gives up (deadline, poison) fails the step.
__device__ __noinline__ void fold_unit(const StepStreamParams& p, int layer, int P, unsigned unit, uint8_t* smem) {
  const DevPhase& ph = p.phases[P];
  uint16_t* y = nullptr;
  unsigned elems = 0;
  const unsigned n = fold_units(p, P);
  const StepFoldDev f = fold_handle(p, P);
  if (!fold_source(p, P, &y, &elems) || n == 0 || f.gen == 0 || f.cell == nullptr || f.recs == nullptr) {
    if (threadIdx.x == 0) atomicCAS(&p.ctl->status, kStepOk, kStepBadUnit);
    return;
  }
  net::bus_write::FoldSmem& sm = *reinterpret_cast<net::bus_write::FoldSmem*>(smem);
  if (threadIdx.x == 0) {
    for (int q = 0; q < static_cast<int>(net::kBusMaxPeers); ++q) sm.peer_done[q] = 0;
    sm.abort = 0;
    sm.gate_waits = 0;
    sm.gate_spins = 0;
  }
  __syncthreads();
  const unsigned long long start = clock64();
  const bool ok = net::bus_write::fold_block(static_cast<int>(unit), static_cast<int>(n), p.fold.view, p.fold.my_rank, f.gen, y, y, elems,
                                            f.cell, f.recs[unit], p.fold.deadline_cycles, start, sm, p.ctl->fold_epoch);
  if (!ok) {
    if (threadIdx.x == 0) atomicCAS(&p.ctl->status, kStepOk, kStepDeadline | (static_cast<unsigned>(P) << 8));
    return;
  }
  // The update on the folded elements of this unit's chunks (the write grid's warp decomposition:
  // chunk c = gwarp + k * nwarps), eight elements a lane.
  {
    constexpr int kWarps = net::bus_write::kThreads / 32;
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t gwarp = unit * kWarps + warp;
    const uint32_t nwarps = n * kWarps;
    const uint32_t chunks = (elems + net::kBusWriteChunkElems - 1) / net::kBusWriteChunkElems;
    const int H = p.hidden;
    const uint16_t* in = ph.accumulate ? p.streams[1] : p.streams[0];
    uint16_t* out = ph.accumulate ? p.streams[0] : p.streams[1];
    if (lane < net::kBusWriteDataLanes)
      for (uint32_t c = gwarp; c < chunks; c += nwarps) {
        const uint32_t e0 = c * net::kBusWriteChunkElems + lane * 8;
        for (uint32_t e = e0; e < e0 + 8 && e < elems; ++e) {
          const int t = static_cast<int>(e / static_cast<unsigned>(H)), d = static_cast<int>(e % static_cast<unsigned>(H));
          mhc_update_elem(p.post, p.comb, bf16_bits_to_float(y[e]), in, out, H, static_cast<size_t>(t), d);
        }
      }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned before = atom_add_acq_rel(&p.ctl->fold_folded[P], 1u);
    if (before + 1 == n) {
      st_rel(&p.ctl->fold_done[P], 1u);  // this group's gate
      red_release(&p.ctl->open_gen, 1u);  // lead units yield: the classes behind the fold want the workers
      p.ctl->stamp_end[P] = gtimer();
      for (int Q = P + 1; Q <= P + static_cast<int>(ph.merged); ++Q) p.ctl->stamp_begin[Q] = p.ctl->stamp_end[P];
    }
  }
  (void)layer;
}

// The attention's finish for one (row, head) (kernels/csa2_attn_finish.cuh
// attn_finish_body): the main source's group partials, then the window's,
// then the sink (the layer's attn_sink range), normalized, rounded, the
// query's rotation removed from the last 64; o's row. The reduce class of the
// phase's last attention pool, gated on every attention pool's blocks.
__device__ __noinline__ void attention_finish_unit(const StepStreamParams& p, int layer, const DevPhase& ph, const DevPool& pool, unsigned unit,
                                                   uint8_t* smem) {
  const StepAttnParams& a = p.attn;
  const int lh = a.local_heads;
  const int r = static_cast<int>(unit) / lh, h = static_cast<int>(unit) % lh;
  const DevPool* src[kStepAttnSources] = {nullptr, nullptr};
  for (int i = 0; i < ph.n_pools; ++i) {
    const DevPool& q = p.pools[ph.pools[i]];
    if (q.kind == kPoolAttention && q.site < kStepAttnSources) src[q.site] = &q;
  }
  const float* m[kStepAttnSources];
  const float* l[kStepAttnSources];
  const float* c[kStepAttnSources];
  int n[kStepAttnSources];
  for (int i = 0; i < kStepAttnSources; ++i) {
    if (src[i] == nullptr) { m[i] = l[i] = c[i] = nullptr; n[i] = 0; continue; }
    const bool grouped = src[i]->slabs == kAttnSplitGroup;
    n[i] = grouped ? src[i]->rows / kAttnSplitGroup : src[i]->rows;
    m[i] = grouped ? a.mg[i] : a.m[i];
    l[i] = grouped ? a.lg[i] : a.l[i];
    c[i] = grouped ? a.cg[i] : a.c[i];
  }
  csa2_finish::FinishSmem& sm = *reinterpret_cast<csa2_finish::FinishSmem*>(smem);
  const float* sink = reinterpret_cast<const float*>(p.bases[layer] + pool.offsets[0]);
  const float* inv_freq = a.inv_freq ? a.inv_freq[layer] : nullptr;
  if (a.o == nullptr || inv_freq == nullptr || a.pos == nullptr) {
    if (threadIdx.x == 0) atomicCAS(&p.ctl->status, kStepOk, kStepBadUnit);
    return;
  }
  csa2_finish::attn_finish_body(r, h, sm, m[1], l[1], c[1], n[1], m[0], l[0], c[0], n[0], sink, lh, a.pos, inv_freq, a.o);
}

// A slabbed projection's reduce unit: chunk `unit` of its [slots, rows]
// outputs as the sum of the slab partials in slab order, rounded once.
// With `update_site`, the attention site's stream update follows on the
// chunk: the pool's y (the output projection, [token][hidden]) is h, and
// streams[0] mix into streams[1] under the site's post and comb.
__device__ void reduce_unit(const StepStreamParams& p, const DevPool& pool, unsigned unit, bool update) {
  const int rows = static_cast<int>(p.ctl->rows), N = pool.rows;
  const int i0 = static_cast<int>(unit) * kStepAccChunk, i1 = min(i0 + kStepAccChunk, rows * N);
  const float* __restrict__ part = p.proj_part + pool.part_offset;
  uint16_t* __restrict__ y = p.proj + pool.out_offset;
  const size_t slab_stride = static_cast<size_t>(kStepMaxRows) * N;
  constexpr int kRound = 4, kMaxSlabs = 8;
  for (int i = i0 + static_cast<int>(threadIdx.x); i < i1; i += kRound * kStepThreads) {
    // Slot t is token t for a projection: y [t][N] is dense, i its index. Four outputs' slabs loaded, then summed in slab order.
    float v[kRound][kMaxSlabs];
#pragma unroll
    for (int r = 0; r < kRound; ++r) {
      const int ir = i + r * kStepThreads;
#pragma unroll
      for (int sl = 0; sl < kMaxSlabs; ++sl) v[r][sl] = (ir < i1 && sl < pool.slabs) ? part[static_cast<size_t>(sl) * slab_stride + static_cast<size_t>(ir)] : 0.f;
    }
#pragma unroll
    for (int r = 0; r < kRound; ++r) {
      const int ir = i + r * kStepThreads;
      if (ir >= i1) break;
      float a = 0.f;
#pragma unroll
      for (int sl = 0; sl < kMaxSlabs; ++sl)
        if (sl < pool.slabs) a += v[r][sl];
      y[ir] = f32_to_bf16(a);
    }
  }
  if (pool.update_site && update) mhc_update_chunk(p.post, p.comb, y, p.streams[0], p.streams[1], N, i0, i1);  // the caller withholds it when the group's fold units carry it
}

// A consumer pool's unit: group g's 128-row block rb of the expert's pair or
// down matrix. `stage`: the group's activations are not yet in shared memory.
// The shared expert's pools have one group, every token's shared slot.
// The fp8 projection form at the shapes compiled in (DeepSeek-V4.1 at world 4).
__device__ bool fp8_matrix_dispatch(int k, int n, const uint8_t* w, const float* scales, const Mxfp4Group& g, int rb, const uint16_t* x,
                                    uint16_t* y, float* partial, uint8_t* smem, int w0, int w1, int unit_rows, bool split, bool stage) {
#define DGPP_STEP_FP8_SHAPE(K, N)                                                                               \
  if (k == (K) && n == (N)) {                                                                                   \
    if (split && (K) <= kSplitMaxK)                                                                             \
      fp8_matrix_split_unit<(K), (N), kConsumerFp8Block, kSplitRowTiles>(w, scales, g, rb, x, y, smem, stage);  \
    else                                                                                                        \
      fp8_matrix<(K), (N), kConsumerFp8Block>(w, scales, g, rb, x, y, partial, smem, w0, w1, unit_rows);       \
    return true;                                                                                                \
  }
  DGPP_STEP_FP8_SHAPE(5120, 1280)   // wq_a
  DGPP_STEP_FP8_SHAPE(5120, 512)    // wkv
  DGPP_STEP_FP8_SHAPE(1280, 8192)   // wq_b
  DGPP_STEP_FP8_SHAPE(4096, 2048)   // wo_a
  DGPP_STEP_FP8_SHAPE(2048, 5120)   // wo_b
  DGPP_STEP_FP8_SHAPE(1280, 4096)   // the indexer's wq_b
  DGPP_STEP_FP8_SHAPE(1536, 25600)  // the Engram projection
  DGPP_STEP_FP8_SHAPE(15360, 5120)  // the draft's main_proj
#undef DGPP_STEP_FP8_SHAPE
  return false;
}

// The production router's select (kernels/glm_moe.cu router_select_warp), one
// warp per token: top_k rounds of argmax over the biased scores, strict > and
// the lower id on ties, the ids sorted ascending, the weights normalized with
// per-element division and scaled. keys[t * top_k + i] takes the ids, the
// control block's slot_weight the weights, 1 for the shared slot.
__device__ void route_select(const StepStreamParams& p, int layer, int* keys, uint8_t* smem) {
  const int rows = static_cast<int>(p.ctl->rows), top_k = static_cast<int>(p.ctl->top_k), spt = top_k + 1, E = p.n_experts;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  // The router units' candidates: units x top_k a token, each unit's top-k over its 16 rows.
  // The merge applies the same rule, so the k picks and their order are the production
  // select's over all E scores: a global top-k element is within its unit's top-k.
  const int units = (E + kRouterUnitRows - 1) / kRouterUnitRows;
  const int n = units * top_k;
  constexpr int kMaxPer = 6;  // candidates a lane: 192 a token at most (24 units by top-k 8, or 32 by 6)
  if (n > 32 * kMaxPer) {
    if (threadIdx.x == 0) atomicCAS(&p.ctl->status, kStepOk, kStepBadUnit);
    return;
  }
  for (int t0 = 0; t0 < rows; t0 += 8) {
    __syncthreads();
    const int t = t0 + warp;
    if (t < rows) {
      int cid[kMaxPer];
      float cb[kMaxPer], cs[kMaxPer];
#pragma unroll
      for (int i = 0; i < kMaxPer; ++i) {
        const int c = lane + i * 32;
        const size_t at = static_cast<size_t>(t) * E + c;
        cid[i] = c < n ? __ldcg(p.cand_id + at) : -1;
        cb[i] = c < n ? __ldcg(p.cand_biased + at) : -INFINITY;
        cs[i] = c < n ? __ldcg(p.cand_score + at) : 0.f;
      }
      int* sel = reinterpret_cast<int*>(smem) + warp * 16;  // [8][16]
      float* wsel = reinterpret_cast<float*>(smem + 8 * 16 * 4) + warp * 16;
      for (int r = 0; r < top_k; ++r) {
        int best = INT_MAX;
        float bv = -INFINITY;
#pragma unroll
        for (int i = 0; i < kMaxPer; ++i)
          if (cid[i] >= 0 && (cb[i] > bv || (cb[i] == bv && cid[i] < best))) { bv = cb[i]; best = cid[i]; }
        router_argmax_lowest_id(bv, best);
        float ws = -INFINITY;
#pragma unroll
        for (int i = 0; i < kMaxPer; ++i)
          if (cid[i] == best) { ws = cs[i]; cb[i] = -INFINITY; cid[i] = -1; }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) ws = fmaxf(ws, __shfl_xor_sync(0xFFFFFFFFu, ws, off));
        if (lane == 0) { sel[r] = best; wsel[r] = ws; }
      }
      __syncwarp();
      if (lane == 0) {
        for (int i = 1; i < top_k; ++i) {
          const int id = sel[i];
          const float wv = wsel[i];
          int j = i - 1;
          while (j >= 0 && sel[j] > id) { sel[j + 1] = sel[j]; wsel[j + 1] = wsel[j]; --j; }
          sel[j + 1] = id;
          wsel[j + 1] = wv;
        }
        float denom = 0.f;
        for (int i = 0; i < top_k; ++i) denom = __fadd_rn(denom, wsel[i]);
        denom = __fadd_rn(denom, 1e-20f);
        for (int i = 0; i < top_k; ++i) {
          const float wv = p.norm_topk ? __fdiv_rn(wsel[i], denom) * p.routed_scaling : wsel[i] * p.routed_scaling;
          keys[t * top_k + i] = sel[i];
          p.ctl->slot_weight[layer][t * spt + i] = wv;
        }
        p.ctl->slot_weight[layer][t * spt + top_k] = 1.0f;
      }
    }
  }
  __syncthreads();
}


// Inlined, with every unit entry: out of line (__noinline__ on the units, the select and the finish)
// the kernel lost 2.5 points on both forms over two samples (2026-11-01), the expert units' hot
// loops paying the call frames and the ABI's register reservation; the router class in the kernel
// was unchanged, so its inflation over the standalone probe (13 µs a wave with the candidate scan)
// is the stream's contention and the hand-out's stagger, not register pressure.
// A row tail (kPoolRowTail): row `row` of its source pool's y through the bodies the production
// decode walk runs one block a row (kernels/csa2_norm.cuh rmsnorm_row and rope_row,
// dsa_latent_append.cuh): the norm with the pool's weight, the rotation of its heads at the row's
// position, and for the kv row its slot list and fp8-block append into the window ring. The
// result goes to this pool's own y (the production writes in place; here each stage's output
// stays for the checks, and the pools behind read this pool's y).
__device__ __forceinline__ void row_tail_unit(const StepStreamParams& p, int layer, const DevPool& pool, const uint8_t* base, int row, uint8_t* smem) {
  const DevPool& sp = p.pools[pool.in_pool];
  const uint16_t* in = p.proj + sp.out_offset + static_cast<size_t>(row) * sp.rows;
  uint16_t* out = p.proj + pool.out_offset + static_cast<size_t>(row) * pool.rows;
  const StepAttnParams& a = p.attn;
  const bool rope = pool.rope_heads > 0;
  if (((rope || pool.append) && a.pos == nullptr) || (rope && (a.inv_freq == nullptr || a.inv_freq[layer] == nullptr)) ||
      (pool.append && (a.ring == nullptr || a.slots == nullptr || a.wlist == nullptr || a.wcounts == nullptr || a.ring_table == nullptr ||
                       a.req_ids == nullptr || a.ring_slots <= 0 || a.window <= 0))) {
    if (threadIdx.x == 0) atomicCAS(&p.ctl->status, kStepOk, kStepBadUnit);
    return;
  }
  if (sp.kind == kPoolFp8Matrix && sp.reduce_by_tail && sp.slabs > 1 && !projection_split(p, sp)) {
    // The source's slab partials summed in slab order (reduce_unit's) into its y, which the norm reads.
    const float* __restrict__ part = p.proj_part + sp.part_offset + static_cast<size_t>(row) * sp.rows;
    const size_t slab_stride = static_cast<size_t>(kStepMaxRows) * sp.rows;
    uint16_t* sy = p.proj + sp.out_offset + static_cast<size_t>(row) * sp.rows;
    for (int i = threadIdx.x; i < sp.rows; i += kStepThreads) {
      float a = 0.f;
      for (int sl = 0; sl < sp.slabs; ++sl) a += part[static_cast<size_t>(sl) * slab_stride + i];
      sy[i] = f32_to_bf16(a);
    }
    __syncthreads();
  }
  if (pool.k > 0) csa2_norm::rmsnorm_row(in, reinterpret_cast<const uint16_t*>(base + pool.offsets[0]), out, pool.k, p.norm_eps, reinterpret_cast<float*>(smem));
  else
    for (int i = threadIdx.x; i < pool.rows; i += kStepThreads) out[i] = in[i];
  if (rope) {
    __syncthreads();  // the row is written before its rotation
    csa2_norm::rope_row(out, out, pool.rope_heads, pool.rope_head_stride, pool.rope_off, kAttnRope / 2, a.pos[row], a.inv_freq[layer]);
  }
  if (pool.append) {
    dsa_tail::decode_tail_slots(row, a.pos[row], a.slots, a.ring_slots, a.window, a.wlist, a.wcounts, nullptr, 0);
    __syncthreads();  // the rotated row and its slot are written for the append
    dsa_tail::latent_append_fp8block_row(p.proj + pool.out_offset, a.req_ids, a.slots, a.ring_table, 1, a.ring_slots,
                                         a.ring + static_cast<unsigned long long>(layer) * a.ring_layer_bytes, kAttnLatent,
                                         latent_row_bytes(LatentFormat::kFp8Block, kAttnLatent), nullptr, 0, row);
  }
}

// A q projection unit's epilogue: the rows' heads rotated over the unit's own output block (the
// decode tail's q rope; the lowering gives the pool one slab, so the block's y is final once every
// warp has written it). Out of line: the production pair math's cosf and sinf keep a local array
// for their range reduction, which inlined grew the kernel's frame (160 to 192 bytes, 2026-11-01).
__device__ __forceinline__ void projection_epilogue_rope(const StepStreamParams& p, int layer, const DevPool& pool, int rows, int block, bool split) {
  if (p.attn.pos == nullptr || p.attn.inv_freq == nullptr || p.attn.inv_freq[layer] == nullptr) {
    if (threadIdx.x == 0) atomicCAS(&p.ctl->status, kStepOk, kStepBadUnit);
    return;
  }
  __syncthreads();
  const int unit_rows = split ? kSplitUnitRows : pool.unit_rows;
  const int n0 = block * unit_rows, n1 = min(n0 + unit_rows, pool.rows);
  csa2_norm::rope_block(p.proj + pool.out_offset, pool.rows, rows, n0, n1, pool.rope_head_stride, pool.rope_off, kAttnRope / 2, p.attn.pos, p.attn.inv_freq[layer]);
}

// The latency units and the q projections' epilogue behind one call: out of line, so their code
// and the production bodies they run (the flash body, the site, the norms and rotation, the
// append) stay out of the worker loop's register allocation, and through one call site, since
// each out-of-line call site adds its own spill and restore code to the loop (three sites: 160 to
// 192 bytes of frame and 42 to 60 local loads, 2026-11-01; the expert units pay for every spill).
__device__ __noinline__ void out_of_line_unit(const StepStreamParams& p, int layer, const DevPool& pool, const uint8_t* base, int rb, uint8_t* smem,
                                              bool epilogue) {
  if (epilogue) {
    projection_epilogue_rope(p, layer, pool, static_cast<int>(p.ctl->rows), projection_split(p, pool) ? rb : rb / pool.slabs, projection_split(p, pool));
    return;
  }
  if (pool.kind == kPoolAttention) {
    attention_unit(p, layer, pool, rb, smem, pool.in_src == 1 ? p.proj + p.pools[pool.in_pool].out_offset : p.attn.q);
    return;
  }
  if (pool.kind == kPoolMhc) {
    mhc_unit(p, layer, pool, base, rb, smem,
             p.unit_trace ? p.unit_trace + (static_cast<size_t>(kStepMaxWorkers) + blockIdx.x % kStepMaxWorkers) * kMhcTraceWords : nullptr);
    return;
  }
  row_tail_unit(p, layer, pool, base, rb, smem);
}

__device__ void consumer_unit(const StepStreamParams& p, int layer, const DevPool& pool, int g, int rb, uint8_t* smem, bool stage) {
  const bool routed = pool.group_kind == kGroupRouted;
  const int expert = routed ? p.ctl->run_list[layer][g] : 0;
  const uint8_t* base = p.bases[layer] + pool.base + static_cast<unsigned long long>(expert) * pool.stride;
  Mxfp4Group grp;
  grp.x_stride = p.x_stride;
  if (routed) {
    grp.slots = p.ctl->group_slots[layer][g];
    grp.tokens = p.ctl->group_tokens[layer][g];
    grp.count = p.ctl->group_count[layer][g];
  } else if (pool.group_kind == kGroupShared) {
    grp.slots = p.ctl->shared_slots[layer];
    grp.tokens = p.ctl->shared_tokens[layer];
    grp.count = static_cast<int>(p.ctl->rows);
  } else {
    grp.slots = p.ctl->token_slots;
    grp.tokens = p.ctl->token_slots;
    grp.count = static_cast<int>(p.ctl->rows);
  }
  if (pool.kind == kPoolAttention || pool.kind == kPoolRowTail || pool.kind == kPoolMhc) {
    out_of_line_unit(p, layer, pool, base, rb, smem, false);
    return;
  }
  if (pool.kind == kPoolRouter) {
    if (pool.k == 5120) {
      // One form compiled: the pipelined batches, instantiated beside this one behind a runtime
      // switch, raised the kernel's stack from 32 to 160 bytes a thread (2026-10-31); the probe
      // measures that form standalone (router_unit_rate).
      const int top_k = static_cast<int>(p.ctl->top_k);
      router_unit<5120, kRouterBatch, kRouterSpecialize, kRouterPipeline>(base, pool, rb, grp, p.x, p.scores, p.biased, p.n_experts, p.router_mode, smem, top_k,
                                                                          p.cand_id, p.cand_score, p.cand_biased,
                                                                          p.unit_trace ? p.unit_trace + (blockIdx.x % kStepMaxWorkers) * kRouterTraceWords : nullptr);
    }
    else if (threadIdx.x == 0) atomicCAS(&p.ctl->status, kStepOk, kStepBadUnit);
    return;
  }
  if (pool.kind == kPoolFp8Matrix) {
    const bool split = projection_split(p, pool);
    // Unit v: a 32-row block in the split form; else rb * slabs + slab for the compute units, then the reduce units.
    const int windows = pool.k / kMxfp4WindowCodes, per_slab = windows / pool.slabs;
    const int slab = split ? 0 : rb % pool.slabs, block = split ? rb : rb / pool.slabs;
    const int w0 = slab * per_slab, w1 = w0 + per_slab;
    float* partial = (!split && pool.slabs > 1) ? p.proj_part + pool.part_offset + static_cast<size_t>(slab) * kStepMaxRows * pool.rows : nullptr;
    // The chain's activations (device_plan.hpp in_src): the step's x rows, a pool's y at its row
    // width, or the attention's o; a block-diagonal matrix reads its head group's columns.
    const uint16_t* xin = p.x;
    if (pool.in_src == 1) {
      const DevPool& sp = p.pools[pool.in_pool];
      xin = p.proj + sp.out_offset;
      grp.x_stride = sp.rows;
    } else if (pool.in_src == 2) {
      xin = p.attn.o;
      grp.x_stride = p.attn.local_heads * kAttnLatent;
    }
    if (pool.in_group_rows) xin += static_cast<size_t>(projection_diag_group(p, pool, rb)) * pool.in_group_cols;
    if (!fp8_matrix_dispatch(pool.k, pool.rows, base + pool.offsets[0], reinterpret_cast<const float*>(base + pool.offsets[1]), grp, block,
                             xin, p.proj + pool.out_offset, partial, smem, w0, w1, pool.unit_rows, split, stage)) {
      if (threadIdx.x == 0) atomicCAS(&p.ctl->status, kStepOk, kStepBadUnit);
      return;
    }
    if (pool.rope_heads) out_of_line_unit(p, layer, pool, base, rb, smem, true);
    return;
  }
  if (pool.kind == kPoolFp8GateUp) {
    Fp8Expert f;
    f.gate = base + pool.offsets[0];
    f.gate_scales = reinterpret_cast<const float*>(base + pool.offsets[1]);
    f.up = base + pool.offsets[2];
    f.up_scales = reinterpret_cast<const float*>(base + pool.offsets[3]);
    fp8_gate_up<kConsumerH, kConsumerI, kConsumerFp8Block>(f, grp, rb, p.x, p.act, smem);
    return;
  }
  if (pool.kind == kPoolFp8Down) {
    Fp8Expert f;
    f.down = base + pool.offsets[0];
    f.down_scales = reinterpret_cast<const float*>(base + pool.offsets[1]);
    if (stage) mxfp4_stage_down_act<kConsumerI>(p.act, grp, smem);
    __syncthreads();
    fp8_down<kConsumerH, kConsumerI, kConsumerFp8Block>(f, grp, rb, smem, p.out, OutIndex{p.hidden, static_cast<int>(p.ctl->top_k) + 1});
    return;
  }
  Mxfp4Expert e;
  if (pool.kind == kPoolMxfp4GateUp || pool.kind == kPoolMxfp4RowGateUp) {
    e.gate = base + pool.offsets[0];
    e.gate_scales = base + pool.offsets[1];
    e.up = base + pool.offsets[2];
    e.up_scales = base + pool.offsets[3];
    if (pool.kind == kPoolMxfp4GateUp) mxfp4_gate_up<kConsumerH, kConsumerI>(e, grp, rb, p.x, p.act, smem);
    else mxfp4_row_gate_up<kConsumerH, kConsumerI>(e, grp, rb, p.x, p.act, smem);
  } else {
    e.down = base + pool.offsets[0];
    e.down_scales = base + pool.offsets[1];
    if (stage) mxfp4_stage_down_act<kConsumerI>(p.act, grp, smem);
    __syncthreads();
    const OutIndex oi{p.hidden, static_cast<int>(p.ctl->top_k) + 1};
    if (pool.kind == kPoolMxfp4Down) mxfp4_down<kConsumerH, kConsumerI>(e, grp, rb, smem, p.out, oi);
    else mxfp4_row_down<kConsumerH, kConsumerI>(e, grp, rb, smem, p.out, oi);
  }
}

__global__ void __launch_bounds__(kStepThreads, 1) step_stream_kernel(StepStreamParams p) {
  StepCtl* ctl = p.ctl;
  const int n_phases = p.layers[p.n_layers - 1].phase_begin + p.layers[p.n_layers - 1].phase_count;
  if (blockIdx.x == 0) {
    // ---- the chain: thread 0 drives the protocol and the phases, the block joins for the router and the tokens
    __shared__ int keys[kStepMaxGroups];
    __shared__ int sorted[kStepMaxGroups];
    extern __shared__ __align__(128) uint8_t chain_smem[];  // the select's score rows
    __shared__ int sm_fail, sm_skip;
    __shared__ int32_t sm_tokens[kStepMaxRows];
    __shared__ unsigned sm_event, sm_selected, sm_first_stamped, sm_fold_pending;
    __shared__ unsigned long long sm_t_begin;
    const bool t0 = threadIdx.x == 0;
    if (t0) { sm_fail = 0; sm_skip = 0; }
    __syncthreads();
    for (int step = 0; step < p.max_steps && !sm_fail && !sm_skip; ++step) {
      const int slot = step & 1;
      volatile StepDescriptor* d = p.desc + slot;
      unsigned long long published = 0;
      // ---- bind: wait for the host to publish execute or skip
      if (t0) {
        const unsigned long long t_wait = gtimer();
        for (;;) {
          const unsigned st = ld_acq_sys(&d->state);
          if (st == kStepExecute) break;
          if (st == kStepSkip) { sm_skip = 1; break; }
          if (ld_acq(&ctl->poison)) { atomicCAS(&ctl->status, kStepOk, kStepPoisoned); sm_fail = 1; break; }
          if (p.pending_deadline_ns && gtimer() - t_wait > p.pending_deadline_ns) {
            atomicCAS(&ctl->status, kStepOk, kStepDeadline | (0xfffu << 8));
            sm_fail = 1;
            break;
          }
          __nanosleep(200);
        }
        published = gtimer();
        if (!sm_fail && !sm_skip) {
          ctl->rows = d->rows;
          ctl->fold_slot = static_cast<unsigned>(slot);  // the step's row of the fold table
          ctl->fold_epoch = d->epoch;                    // into every fold chunk's trailer hash (a stale chunk of an earlier step fails the proof)
          ctl->top_k = d->top_k;
          // The token group's slots are the step's rows from the start: the first layer's router reads them before any route ran.
          for (int i = 0; i < 32; ++i) ctl->token_slots[i] = i;
          ctl->seed = d->seed + static_cast<unsigned>(d->seq);
          ctl->route_pool = p.route_pool;
          ctl->pf_next = 0;
          ctl->pf_bytes = 0;
          ctl->consumed_pf_bytes = 0;
          ctl->open_gen = 0;
        }
      }
      __syncthreads();
      if (!sm_skip && !sm_fail) {
        // The step's counters and trace, reset by the whole block (one thread took tens of microseconds).
        for (int i = static_cast<int>(threadIdx.x); i < n_phases; i += kStepThreads) {
          ctl->done[i] = 0; ctl->pf_units[i] = 0; ctl->idle_first[i] = 0; ctl->fold_done[i] = 0; ctl->fold_staged[i] = 0; ctl->fold_folded[i] = 0;
          ctl->busy_ns[i] = 0; ctl->grabs[i] = 0; ctl->handout_ns[i] = 0; ctl->lead_in_bytes[i] = 0; ctl->pf_credit[i] = 0; ctl->pf_staged[i] = 0;
          ctl->wait_ns[i] = 0;
          for (int c = 0; c < kStepMaxClasses; ++c) { ctl->cls_next[i][c] = 0; ctl->cls_first[i][c] = ~0ull; ctl->cls_last[i][c] = 0; ctl->cls_last_grab[i][c] = 0; }
        }
        for (int i = static_cast<int>(threadIdx.x); i < kStepMaxLayers * n_phases; i += kStepThreads) {
          ctl->pool_done[i / n_phases][i % n_phases] = 0;
          ctl->reduce_done[i / n_phases][i % n_phases] = 0;
        }
        for (int i = static_cast<int>(threadIdx.x); i < kStepMaxLayers; i += kStepThreads) {
          ctl->acc_done[i] = 0; ctl->site_done[i][0] = 0; ctl->site_done[i][1] = 0;
        }
        __syncthreads();
        if (t0) {
          build_class_table(p, 0);  // the first group's classes; each group builds the next one's while it runs
          __threadfence();
          st_rel(&ctl->phase_word, word(step, 0, kStateIdle));
        }
      }
      __syncthreads();
      if (sm_skip || sm_fail) {
        if (t0) {
          volatile StepVerdict* v = p.verdict + slot;
          v->status = sm_skip ? kVerdictSkipped : (ctl->status & 0xff) == kStepPoisoned ? kVerdictPoisoned : kVerdictDeadline;
          v->rows = 0;
          v->epoch = d->epoch;
          v->tokens_fold = 0;
          v->begin_ns = published;
          v->end_ns = gtimer();
          v->published_ns = published;
          __threadfence_system();
          st_rel_sys64(&v->seq, d->seq);
        }
        break;
      }
      // ---- the tokens: the device feed the last step wrote, or the host's
      const int rows = static_cast<int>(ctl->rows);
      const unsigned long long seq = d->seq;
      // The first hyper-connection site's pre is the one-hot, as the model's forward seeds it.
      if (p.pre[0] && threadIdx.x < static_cast<unsigned>(rows) * kMhcN) p.pre[0][threadIdx.x] = (threadIdx.x % kMhcN) == 0 ? 1.f : 0.f;
      if (threadIdx.x < static_cast<unsigned>(rows)) {
        const int32_t tok = d->tokens_from_host ? d->tokens[threadIdx.x] : p.token_feed[slot * kStepMaxRows + threadIdx.x];
        const int32_t next = step_next_token(tok, seq, static_cast<int>(threadIdx.x));
        p.token_feed[(slot ^ 1) * kStepMaxRows + threadIdx.x] = next;
        sm_tokens[threadIdx.x] = next;
      }
      __syncthreads();
      const unsigned long long t_step = gtimer();
      // ---- the phases: each published with its merged group, in order
      for (int P = 0; P < n_phases && !sm_fail;) {
        const DevPhase& ph = p.phases[P];
        const int l = ph.layer;
        const DevLayer& dl = p.layers[l];
        const int merged = static_cast<int>(ph.merged);
        const int P_last = P + merged;
        // The group's facts, resolved by the lowering (one phase record, not a walk of the tables).
        // Its router, if it resolves one: its real pool's units are counted, and the chain selects
        // as soon as they are complete, inside the group, so the next phase's groups are known
        // when the group ends.
        const float fold_us = ph.group_fold_us;
        const int R = ph.group_router_phase, router_pool = ph.group_router_pool, routed_pool = ph.group_routed_pool;
        const unsigned router_units = router_pool >= 0 && routed_pool >= 0 ? pool_compute_units(p, l, p.pools[router_pool]) : 0u;
        (void)dl;
        if (t0) {
          sm_selected = 0;
          const unsigned long long t_begin = gtimer();
          sm_t_begin = t_begin;
          for (int Q = P; Q <= P_last; ++Q) ctl->stamp_begin[Q] = t_begin;
          // A gather phase waits for the host's rows (section 2.11).
          if (ph.waits_host && d->gather_seq) {
            while (ld_acq_sys64(p.host_seq) < d->gather_seq) {
              if (p.deadline_ns && gtimer() - t_begin > p.deadline_ns) {
                atomicCAS(&ctl->status, kStepOk, kStepDeadline | (static_cast<unsigned>(P) << 8));
                sm_fail = 1;
                break;
              }
              if (ld_acq(&ctl->poison)) { atomicCAS(&ctl->status, kStepOk, kStepPoisoned); sm_fail = 1; break; }
            }
          }
          // The lead never takes the units of the group that is running: a stream phase's
          // workers read them now, a latency phase's chain does, and the merged phases' lead-only
          // bytes are staged by the group's own staging class at the fold's trigger (staged by the
          // lead early in the group they were evicted before their classes ran, 2026-11-01).
          for (int Q = P; Q <= P_last; ++Q) atomicMax(&ctl->pf_next, p.phases[Q].lead_begin + p.phases[Q].lead_units);
          {
            const long long ahead = static_cast<long long>(ctl->pf_bytes) - static_cast<long long>(ctl->consumed_pf_bytes);
            ctl->ahead_begin[P] = ahead > 0 ? static_cast<unsigned long long>(ahead) : 0ull;
            const unsigned pf = ld_acq(&ctl->pf_next);
            ctl->cursor_begin[P] = pf;  // the lead unit index; the host maps it to a phase (a binary search over the phase table here cost the chain several microseconds a group)
            ctl->look_blocked[P] = 0;
          }
          sm_first_stamped = 0;
          sm_fold_pending = (ph.kind == 1 && merged > 0 && fold_us > 0.f) ? 1u : 0u;
          if (!sm_fail && ph.kind == 1) st_rel(&ctl->phase_word, word(step, P, kStateStream));
          else if (!sm_fail) {
            // A latency phase is published too, idle, so the workers' lead sees the phase
            // the chain is in: its idle budget and the next stream phase's exemption.
            st_rel(&ctl->phase_word, word(step, P, kStateIdle));
          }
          // The next group's classes, built while this group runs; the routed pools' counts are
          // resolved by the workers against the run list when they load the table. (Built two groups
          // ahead, so a short group would not expose the build behind it, the rate read 71.5 to 73.3
          // against 76.7 to 76.9 and the empty attend phase was no shorter, 2026-11-01: the phase's
          // 15 to 20 µs is the chain's bookkeeping around a group, not the table.)
          if (!sm_fail && P_last + 1 < n_phases) build_class_table(p, P_last + 1);
          if (!sm_fail && ph.kind != 1) spin_us(ph.latency_us);
        }
        __syncthreads();
        if (!sm_fail && ph.kind == 1) {
          // The wait, in rounds: thread 0 polls the group's units and events (the fold's trigger,
          // the router's completion); the block joins for the select.
          const unsigned need = phase_units(p, P);
          for (;;) {
            if (t0) {
              sm_event = 0;
              while (ld_acq(&ctl->done[P]) < need) {
                if (sm_fold_pending && fold_trigger_holds(p, l, P)) {
                  // Lead units yield at the trigger: the merged phases' staging classes open on it and
                  // their 77 units of 64 KB want the workers through the fold's window (left to
                  // finish their lead units, the workers reached the staging 10 to 20 µs late and it
                  // was still running when the router's gate opened, 2026-11-01).
                  red_release(&ctl->open_gen, 1u);
                  if (fold_units(p, P)) {
                    // Past world 1 the fold is the stage and fold classes' (net/bus_write.cuh): the last
                    // fold unit sets fold_done and the stamps; the chain only stops waiting for it.
                    sm_first_stamped = 1;
                  } else {
                    spin_us(fold_us);
                    st_rel(&ctl->fold_done[P], 1u);  // this group's gate (per group, not per layer: a layer has two groups with folds)
                    red_release(&ctl->open_gen, 1u);  // lead units yield: the classes behind the fold want the workers
                    ctl->stamp_end[P] = gtimer();  // the first phase's trace: its own units and the fold
                    for (int Q = P + 1; Q <= P_last; ++Q) ctl->stamp_begin[Q] = ctl->stamp_end[P];
                    sm_first_stamped = 1;
                  }
                  sm_fold_pending = 0;
                }
                if (router_units && !sm_selected && ld_acq(&ctl->pool_done[l][router_pool]) >= router_units) { sm_event = 1; break; }
                if (p.deadline_ns && gtimer() - sm_t_begin > p.deadline_ns) {
                  atomicCAS(&ctl->status, kStepOk, kStepDeadline | (static_cast<unsigned>(P) << 8));
                  sm_fail = 1;
                  break;
                }
                if (ld_acq(&ctl->poison)) { atomicCAS(&ctl->status, kStepOk, kStepPoisoned); sm_fail = 1; break; }
              }
              if (!sm_fail && sm_event == 0) {
                if (sm_fold_pending) {  // no gated class waited on it: the fold after the group
                  spin_us(fold_us);
                  st_rel(&ctl->fold_done[P], 1u);  // this group's gate (per group, not per layer: a layer has two groups with folds)
                  red_release(&ctl->open_gen, 1u);  // lead units yield: the classes behind the fold want the workers
                  sm_fold_pending = 0;
                }
                st_rel(&ctl->phase_word, word(step, P, kStateIdle));
                sm_event = 2;
              }
            }
            __syncthreads();
            if (sm_fail || sm_event == 2) break;
            // The select inside the group: the router's units are complete (thread 0 acquired their
            // count; the barrier carries it to the block, which reads the scores past L1).
            const unsigned long long t_sel = gtimer();
            route_select(p, l, keys, chain_smem);
            const unsigned long long t_fin = gtimer();
            route_finish(p, l, keys, sorted, chain_smem + 2 * 8 * 16 * 4);
            if (t0) {
              ctl->route_select_ns[l] = t_fin - t_sel;
              ctl->route_finish_ns[l] = gtimer() - t_fin;
              sm_selected = 1;
            }
            __syncthreads();
          }
        }
        if (t0 && !sm_fail) {
          // The group's bytes are consumed, the lead's among them: whatever the workers have
          // not credited unit by unit frees up now.
          for (int Q = P; Q <= P_last; ++Q) atomicAdd(&ctl->consumed_pf_bytes, ctl->pf_staged[Q] - ctl->pf_credit[Q]);
          const unsigned long long t_end = gtimer();
          for (int Q = sm_first_stamped ? P + 1 : P; Q <= P_last; ++Q) ctl->stamp_end[Q] = t_end;
        }
        __syncthreads();
        if (sm_fail) break;
        if (R >= 0 && !sm_selected && routed_pool >= 0) {
          // The select after the group: the synthetic picks, or a router whose units ran as a lone phase.
          const unsigned long long t_sel = gtimer();
          if (router_pool >= 0) {
            route_select(p, l, keys, chain_smem);
          } else {
            route_synthetic_picks(p, l, p.pools[routed_pool], keys);
          }
          const unsigned long long t_fin = gtimer();
          route_finish(p, l, keys, sorted, chain_smem + 2 * 8 * 16 * 4);
          if (t0) {
            ctl->route_select_ns[l] = t_fin - t_sel;
            ctl->route_finish_ns[l] = gtimer() - t_fin;
          }
          __syncthreads();
        }
        // A lone phase's fold follows it (a group spun its fold inside, at its trigger).
        if (t0 && merged == 0 && fold_us > 0.f) {
          spin_us(fold_us);
          ctl->stamp_end[P] = gtimer();
        }
        __syncthreads();
        P = P_last + 1;
      }
      // ---- the verdict
      if (t0) {
        volatile StepVerdict* v = p.verdict + slot;
        unsigned long long fold = 0x9e3779b97f4a7c15ull;
        for (int r = 0; r < rows; ++r) fold = step_fold_mix(fold, static_cast<uint64_t>(static_cast<uint32_t>(sm_tokens[r])));
        const unsigned st = ctl->status & 0xffu;
        v->status = sm_fail ? (st == kStepPoisoned ? kVerdictPoisoned : st == kStepBadUnit ? kVerdictBadUnit : kVerdictDeadline) | (ctl->status & ~0xffu)
                            : kVerdictOk;
        v->rows = static_cast<uint32_t>(rows);
        v->epoch = d->epoch;
        v->tokens_fold = fold;
        v->begin_ns = t_step;
        v->end_ns = gtimer();
        v->published_ns = published;
        if (!sm_fail) ++ctl->steps_run;
        __threadfence_system();
        st_rel_sys64(&v->seq, seq);
      }
      __syncthreads();
    }
    if (t0) {
      __threadfence();
      st_rel(&ctl->phase_word, word(0, 0, kStateEnd));
    }
    return;
  }
  // ---- a worker
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ unsigned sm_kind, sm_unit, sm_phase, sm_count, sm_word, sm_stop, sm_published, sm_look_blocked, sm_class, sm_gen, sm_stage_open;
  __shared__ unsigned sm_lead_begin[kStepMaxPhases];  // the lead sequence's phase boundaries, for the lead's hand-out
  __shared__ unsigned long long sm_grab_bytes;
  // The published group's unit classes, built once a group a worker: in hand-out order, each
  // class's unit count, grab size, phase, kind (0 compute, 1 reduce, 2 fixed bytes, 3 accumulate),
  // pool and gate. A class has its own counter (ctl->cls_next), so a closed or exhausted class is
  // skipped and the first open class with units left is taken.
  __shared__ unsigned sm_cache_word, sm_nclasses, sm_cls_exhausted, sm_rows;
  __shared__ unsigned sm_cls_count[kStepMaxClasses], sm_cls_grab[kStepMaxClasses];
  __shared__ int sm_cls_phase[kStepMaxClasses], sm_cls_pool[kStepMaxClasses];
  __shared__ uint8_t sm_cls_kind[kStepMaxClasses], sm_cls_gate[kStepMaxClasses], sm_cls_gate_arg[kStepMaxClasses];
  __shared__ unsigned sm_cls_base[kStepMaxClasses];  // a staging class: its first lead unit in the phase's sequence
  if (threadIdx.x == 0) { sm_cache_word = ~0u; sm_word = ~0u; sm_look_blocked = 0; }
  for (int i = static_cast<int>(threadIdx.x); i < n_phases; i += kStepThreads) sm_lead_begin[i] = p.phases[i].lead_begin;
  __syncthreads();
  for (;;) {
    if (threadIdx.x == 0) {
      sm_kind = 0;
      sm_count = 1;
      const unsigned long long t_top = gtimer();
      const unsigned w = ld_acq(&ctl->phase_word);
      if (w != sm_word) sm_look_blocked = 0;  // a new published phase: the lookahead block is counted once a worker a phase
      sm_word = w;
      sm_published = (w >> 2) & 0xfffffu;
      const unsigned state = w & 3u;
      const int P = static_cast<int>((w >> 2) & 0xfffffu);
      if (state == kStateEnd || ld_acq(&ctl->poison)) {
        sm_kind = 3;
      } else {
        const int cur_layer = p.phases[P].layer;
        if (state == kStateStream) {
          if (w != sm_cache_word) {
            // The group's classes, as the chain built them: one round trip of independent loads.
            sm_rows = ctl->rows;
            const unsigned n = min(ld_acq(&ctl->cls_n[P]), static_cast<unsigned>(kStepMaxClasses));
            uint4 e[kStepMaxClasses];
#pragma unroll
            for (int c = 0; c < kStepMaxClasses; ++c) e[c] = c < static_cast<int>(n) ? __ldcg(&ctl->cls_table[P][c]) : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
            for (int c = 0; c < kStepMaxClasses; ++c) {
              const int cphase = static_cast<int>(e[c].z & 0xffffu);
              sm_cls_count[c] = (e[c].w & 0xffu) ? e[c].x * static_cast<unsigned>(ld_acq(reinterpret_cast<unsigned*>(&ctl->run_count[p.phases[cphase].layer]))) : e[c].x;
              sm_cls_base[c] = e[c].w >> 8;
              sm_cls_grab[c] = e[c].y & 0xffu;
              sm_cls_kind[c] = static_cast<uint8_t>((e[c].y >> 8) & 0xffu);
              sm_cls_gate[c] = static_cast<uint8_t>((e[c].y >> 16) & 0xffu);
              sm_cls_gate_arg[c] = static_cast<uint8_t>(e[c].y >> 24);
              sm_cls_phase[c] = static_cast<int>(e[c].z & 0xffffu);
              sm_cls_pool[c] = static_cast<int>(static_cast<int16_t>(e[c].z >> 16));
            }
            sm_nclasses = n;
            sm_cls_exhausted = 0u;
            sm_cache_word = w;
          }
          // The grab: the first open class with units left. Consecutive units share their
          // staging while the class has units for every worker; the tail takes them singly.
          bool any_pending = false;  // a closed gate: units will come
          sm_stage_open = 0;         // a staging class open with units left: the lead waits for it
          for (unsigned c = 0; c < sm_nclasses && sm_kind == 0; ++c) {
            if (sm_cls_exhausted & (1u << c)) continue;
            const uint8_t gk = sm_cls_gate[c];
            if (gk != kGateNone) {
              const int il = p.phases[sm_cls_phase[c]].layer;
              bool open;
              if (gk == kGateFold) open = ld_acq(&ctl->fold_done[P]) != 0u;  // this group's fold
              else if (gk == kGateSite) open = ld_acq(&ctl->site_done[il][sm_cls_gate_arg[c] & 1u]) >= sm_rows;
              else if (gk == kGateTrigger) open = fold_trigger_holds(p, cur_layer, P);  // the fold's window begins
              else if (gk == kGateAttnPools) open = attn_pools_done(p, il, sm_cls_phase[c]);  // every source's blocks: the finish
              else if (gk == kGateFoldStaged) open = ld_acq(&ctl->fold_staged[P]) >= fold_units(p, P);  // the fold's row staged
              else if (gk == kGatePool) open = pool_gate_open(p, il, p.pools[sm_cls_pool[c]]);  // the chain: the pool whose y the class reads is complete (and its other gate)
              else open = ld_acq(&ctl->pool_done[il][sm_cls_pool[c]]) >= pool_compute_units(p, il, p.pools[sm_cls_pool[c]]);
              if (!open) { any_pending = true; continue; }
            }
            const unsigned count = sm_cls_count[c];
            if (sm_cls_kind[c] == kClassStage && sm_cls_pool[c] >= 0) {
              // A gated pool's staging: once its gate is open its own units read the bytes, so the
              // class closes; the units never handed out are credited as done for the chain's count.
              const int il = p.phases[sm_cls_phase[c]].layer;
              if (pool_gate_open(p, il, p.pools[sm_cls_pool[c]])) {
                const unsigned u = atomicAdd(&ctl->cls_next[P][c], count);
                if (u < count) red_release(&ctl->done[P], count - u);
                sm_cls_exhausted |= 1u << c;
                continue;
              }
            }
            const unsigned at = ld_acq(&ctl->cls_next[P][c]);
            if (at >= count) {
              sm_cls_exhausted |= 1u << c;
              continue;
            }
            const bool staging = sm_cls_kind[c] == kClassStage;
            if (staging) {
              sm_stage_open = 1;
              if (p.stage_workers && atomicAdd(&ctl->stage_active, 1u) >= p.stage_workers) {  // the cap: the memory is saturated already
                atomicSub(&ctl->stage_active, 1u);
                any_pending = true;
                continue;
              }
            }
            unsigned grab = sm_cls_grab[c];
            if (grab > 1) {  // the tail shrinks the grab gradually: at most a worker's share of what remains
              const unsigned share = (count - at) / (gridDim.x - 1);
              grab = share < 1u ? 1u : (share < grab ? share : grab);
            }
            const unsigned u = atomicAdd(&ctl->cls_next[P][c], grab);
            if (u >= count) {
              if (staging && p.stage_workers) atomicSub(&ctl->stage_active, 1u);
              sm_cls_exhausted |= 1u << c;
              continue;
            }
            sm_kind = 1; sm_class = c; sm_unit = u; sm_phase = static_cast<unsigned>(P);
            sm_count = min(grab, count - u);
            if (u == 0) ctl->cls_first[P][c] = t_top;  // the class's first unit: its trace
            atomicMax(&ctl->cls_last_grab[P][c], t_top);
            atomicAdd(&ctl->handout_ns[P], gtimer() - t_top);
          }
          // (Workers leading inside the routed expert phase in place of taking its units were
          // measured 2026-10-31 and lost four to five points: that phase is memory-saturated,
          // so the staging added to it one to one and bought back a quarter of itself in qkv.)
          if (sm_kind == 0 && !any_pending && ctl->idle_first[P] == 0) ctl->idle_first[P] = gtimer();
        }
        if (sm_kind == 0 && p.lead_cap_bytes && !(p.stage_lead_pause && sm_stage_open)) {  // the staging's bytes first: they are on the chain's path
          long long ahead = static_cast<long long>(ctl->pf_bytes) - static_cast<long long>(ctl->consumed_pf_bytes);
          if (ahead < 0) ahead = 0;
          // The idle budget: the modeled time until the next stream phase has units, a latency
          // phase's own time and the folds between, when this phase is a latency phase or drained.
          unsigned long long cap = p.lead_cap_bytes;
          if (p.lead_bytes_per_us > 0.f) {
            float idle_us = 0.f;
            for (int q = P; q < n_phases; ++q) {
              const DevPhase& qh = p.phases[q];
              if (qh.kind == 1 && !(q == P && state == kStateStream) ) break;  // a later stream phase: the memory is busy again
              idle_us += qh.latency_us;  // a latency phase's time, or a compute-bound stream phase's allowance (the sites, the accumulate)
              float fold = qh.fold_after_us;
              for (int m = 1; m <= static_cast<int>(qh.merged); ++m) fold = fmaxf(fold, p.phases[q + m].fold_after_us);
              idle_us += fold;
              q += static_cast<int>(qh.merged);
            }
            cap += static_cast<unsigned long long>(idle_us * p.lead_bytes_per_us);
          }
          if (static_cast<unsigned long long>(ahead) + p.unit_bytes <= cap) {
            bool slot = true;
            if (p.max_leaders && atomicAdd(&ctl->leaders, 1u) >= p.max_leaders) { atomicSub(&ctl->leaders, 1u); slot = false; }  // the governor's bound
            if (slot) {
              const unsigned pf = ld_acq(&ctl->pf_next);
              if (pf < p.lead_units) {
                const int P2 = phase_of_lead_unit_sm(sm_lead_begin, n_phases, pf);
                if (p.phases[P2].layer > cur_layer + p.lookahead_layers) {
                  if (!sm_look_blocked) { sm_look_blocked = 1u; atomicAdd(&ctl->look_blocked[P], 1u); }
                } else {
                  const unsigned got = atomicAdd(&ctl->pf_next, 1u);
                  if (got < p.lead_units) {
                    const int P3 = phase_of_lead_unit_sm(sm_lead_begin, n_phases, got);
                    sm_kind = 2; sm_unit = got - p.phases[P3].lead_begin; sm_phase = static_cast<unsigned>(P3);
                    sm_gen = ld_acq(&ctl->open_gen);
                    atomicAdd(&ctl->pf_bytes, static_cast<unsigned long long>(p.unit_bytes));
                    atomicAdd(&ctl->pf_units[P3], 1u);
                  }
                }
              }
              if (sm_kind != 2 && p.max_leaders) atomicSub(&ctl->leaders, 1u);
            }
          }
        }
      }
    }
    __syncthreads();
    const unsigned kind = sm_kind;
    if (kind == 3) return;
    if (kind == 0) { __nanosleep(200); continue; }
    const unsigned long long t_grab = gtimer();
    if (threadIdx.x == 0) sm_grab_bytes = 0;
    // Each unit of the grab on its own, all of one class: a compute unit of a pool, a reduce
    // unit, a fixed byte unit or an accumulation chunk, of the class's phase and layer (a merged
    // group may span two layers); a lead unit (kind 2) streams a segment of its own phase.
    const unsigned cls = kind == 1 ? sm_class : 0u;
    const int xphase = kind == 1 ? sm_cls_phase[cls] : static_cast<int>(sm_phase);
    const DevPhase& uph = p.phases[xphase];
    const int xlayer = uph.layer;
    const uint8_t ckind = kind == 1 ? sm_cls_kind[cls] : kClassFixed;
    const int pool_id = kind == 1 ? sm_cls_pool[cls] : -1;
    const DevPool* upool = pool_id >= 0 ? &p.pools[pool_id] : nullptr;
    int last_group = -1, last_pool = -1;
    for (unsigned q = 0; q < sm_count; ++q) {
      const unsigned xunit = sm_unit + q;  // the unit's index inside its class
      if (q) __syncthreads();  // the previous unit's readers of the shared memory are done
      if (kind == 1 && ckind == kClassStage) {
        // A staging unit streams whole into L2 (evict_last). Yielding when a gate opens, as a lead
        // unit does, was measured 2026-11-01 and lost a point: the router's units then read the
        // rows the yield left unstaged from DRAM (their class 23 µs where it had been 18), which
        // cost more than the hand-out's stagger saved.
        const uint8_t* ptr = nullptr;
        unsigned bytes = 0;
        unsigned acc = 0;
        if (locate_lead(p, xlayer, xphase, sm_cls_base[cls] + xunit, &ptr, &bytes)) acc = stream_unit<false>(ptr, bytes);
        else if (threadIdx.x == 0) atomicCAS(&ctl->status, kStepOk, kStepBadUnit);
        for (int o = 16; o; o >>= 1) acc ^= __shfl_xor_sync(0xffffffffu, acc, o);
        if ((threadIdx.x & 31) == 0) p.sink[(blockIdx.x % kStepMaxWorkers) * 8 + (threadIdx.x >> 5)] ^= acc;
        if (threadIdx.x == 0 && q + 1 == sm_count && p.stage_workers) atomicSub(&ctl->stage_active, 1u);  // the grab's last unit: the cap's slot back
        continue;
      }
      if (kind == 1 && ckind == kClassFoldStage) {
        fold_stage_unit(p, xphase, xunit);
        continue;
      }
      if (kind == 1 && ckind == kClassFold) {
        fold_unit(p, xlayer, xphase, xunit, smem);
        continue;
      }
      if (kind == 1 && ckind == kClassAccumulate) {
        accumulate_chunk(p, xlayer, xunit, uph.update != 0 && fold_units(p, xphase) == 0u);  // the group's fold units apply the update past world 1
        __syncthreads();
        if (threadIdx.x == 0) red_release(&ctl->acc_done[xlayer], 1u);
        continue;
      }
      if (kind == 1 && ckind == kClassReduce) {
        // A reduce unit: its class opened on the pool's compute units; the wait is the fallback.
        const unsigned compute_units = pool_compute_units(p, xlayer, *upool);
        if (threadIdx.x == 0) {
          const unsigned long long t_wait = gtimer();
          while (ld_acq(&ctl->pool_done[xlayer][pool_id]) < compute_units) {
            if (p.deadline_ns && gtimer() - t_wait > p.deadline_ns) { atomicCAS(&ctl->status, kStepOk, kStepDeadline | (static_cast<unsigned>(xphase) << 8)); break; }
            if (ld_acq(&ctl->poison)) break;
            __nanosleep(100);
          }
        }
        __syncthreads();
        if (upool->kind == kPoolAttention) attention_finish_unit(p, xlayer, uph, *upool, xunit, smem);
        else reduce_unit(p, *upool, xunit, fold_units(p, xphase) == 0u);  // the group's fold units apply the update past world 1
        __syncthreads();
        if (threadIdx.x == 0) red_release(&ctl->reduce_done[xlayer][pool_id], 1u);
        last_group = -1;
        last_pool = -1;
        continue;
      }
      if (kind == 1 && ckind == kClassCompute && upool->kind != kPoolBytes) {
        const bool shared = upool->group_kind == kGroupShared;
        const unsigned upg = compute_units_per_group(p, *upool);
        const int g = static_cast<int>(xunit / upg);
        const int rb = static_cast<int>(xunit % upg);
        if (pool_is_down(upool->kind) && upool->partner >= 0) {
          // The group's gate and up units must be complete: they were handed out earlier, so the wait is short.
          if (threadIdx.x == 0) {
            const unsigned gate_units = static_cast<unsigned>(p.pools[upool->partner].units_per_group);
            const unsigned* done = shared ? &ctl->shared_gate_done[xlayer] : &ctl->gate_done[xlayer][g];
            const unsigned long long t_wait = gtimer();
            while (ld_acq(done) < gate_units) {
              if (p.deadline_ns && gtimer() - t_wait > p.deadline_ns) { atomicCAS(&ctl->status, kStepOk, kStepDeadline | (static_cast<unsigned>(xphase) << 8)); break; }
              if (ld_acq(&ctl->poison)) break;
              __nanosleep(100);
            }
            atomicAdd(&ctl->wait_ns[sm_phase], gtimer() - t_wait);
          }
          __syncthreads();
        }
        // Consecutive units of one group share their staged activations; a block-diagonal
        // projection's units share them within a head group.
        const int key = (upool->kind == kPoolFp8Matrix && upool->in_group_rows) ? projection_diag_group(p, *upool, rb) : g;
        const unsigned long long t_unit = gtimer();
        consumer_unit(p, xlayer, *upool, g, rb, smem, pool_id != last_pool || key != last_group);
        last_group = key;
        last_pool = pool_id;
        if (threadIdx.x == 0) sm_grab_bytes += upool->bytes / upg;
        if (upool->kind == kPoolFp8Matrix || upool->kind == kPoolRouter || upool->kind == kPoolAttention || upool->kind == kPoolRowTail) {
          __syncthreads();  // the pool's compute units complete: its reduce units' gate, the chain's select, or the pools behind it (kGatePool)
          if (threadIdx.x == 0) {
            const unsigned before = atom_add_acq_rel(&ctl->pool_done[xlayer][pool_id], 1u);
            if (before + 1 == pool_compute_units(p, xlayer, *upool)) red_release(&ctl->open_gen, 1u);  // the last: a gate opens, lead units yield
            if (p.unit_trace && pool_id == p.trace_pool) {  // the traced pool: this worker's unit durations (bank 2)
              unsigned long long* tr = p.unit_trace + (2u * kStepMaxWorkers + blockIdx.x % kStepMaxWorkers) * 8;
              const unsigned long long now = gtimer(), d = now - t_unit;
              tr[0] += 1;
              tr[1] += d;
              tr[2] = d > tr[2] ? d : tr[2];
              tr[3] = tr[3] == 0 || d < tr[3] ? d : tr[3];
              if (tr[4] == 0) tr[4] = t_unit;
              tr[5] = now;
            }
          }
        }
        if (pool_is_gate_up(upool->kind)) {
          __syncthreads();
          if (threadIdx.x == 0) red_release(shared ? &ctl->shared_gate_done[xlayer] : &ctl->gate_done[xlayer][g], 1u);
        }
      } else {
        last_group = -1;
        last_pool = -1;
        const uint8_t* ptr = nullptr;
        unsigned bytes = 0;
        unsigned acc = 0;
        const bool found = kind == 2  ? locate_lead(p, xlayer, xphase, xunit, &ptr, &bytes)
                           : upool    ? locate_pool_bytes(p, xlayer, *upool, xunit, &ptr, &bytes)
                                      : locate_fixed(p, xlayer, xphase, xunit, &ptr, &bytes);
        if (found && kind == 2) {
          // A lead unit yields: it streams in pieces and stops when the chain publishes a new
          // phase, so a worker is back at the hand-out within a piece's time rather than a
          // unit's at the memory's share, which held phase starts behind lead work.
          unsigned off = 0;
          while (off < bytes) {
            const unsigned piece = min(kLeadPiece, bytes - off);
            acc ^= stream_unit<false>(ptr + off, piece);
            off += piece;
            if (off < bytes) {
              if (threadIdx.x == 0) {
                const unsigned now = ld_acq(&ctl->phase_word);
                // A stream phase with units, or an opened gate, wants the worker (a 64 KB lead unit at a
                // 47-way share of the memory is about 13 µs; the merged classes' units waited that long).
                sm_stop = ((now != sm_word && (now & 3u) == kStateStream) || ld_acq(&ctl->open_gen) != sm_gen) ? 1u : 0u;
              }
              __syncthreads();
              if (sm_stop) break;
            }
          }
          if (threadIdx.x == 0) {
            if (off < bytes)  // the unstreamed remainder leaves the lead's account (a wrapping add)
              atomicAdd(&ctl->pf_bytes, static_cast<unsigned long long>(-static_cast<long long>(bytes - off)));
            atomicAdd(&ctl->pf_staged[sm_phase], static_cast<unsigned long long>(off));
            atomicAdd(&ctl->lead_in_bytes[sm_published], static_cast<unsigned long long>(off));
          }
        } else if (found) {
          acc = stream_unit<true>(ptr, bytes);
          if (threadIdx.x == 0) sm_grab_bytes += bytes;
        } else if (threadIdx.x == 0) {
          atomicCAS(&ctl->status, kStepOk, kStepBadUnit);
        }
        for (int o = 16; o; o >>= 1) acc ^= __shfl_xor_sync(0xffffffffu, acc, o);
        if ((threadIdx.x & 31) == 0) p.sink[(blockIdx.x % kStepMaxWorkers) * 8 + (threadIdx.x >> 5)] ^= acc;
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      if (kind == 1) {
        // The staged bytes this grab consumed leave the lead's account now rather than at the
        // phase's end, so the lead's room opens as the phase runs and the staged set in L2
        // stays within the cap, in consumption order.
        const unsigned long long staged = ctl->pf_staged[sm_phase];
        const unsigned long long credited = ctl->pf_credit[sm_phase];
        if (credited < staged) {
          const unsigned long long room = staged - credited;
          const unsigned long long credit = sm_grab_bytes < room ? sm_grab_bytes : room;
          atomicAdd(&ctl->pf_credit[sm_phase], credit);
          atomicAdd(&ctl->consumed_pf_bytes, credit);
        }
        red_release(&ctl->done[sm_phase], sm_count);
        atomicMax(&ctl->cls_last[sm_phase][sm_class], gtimer());
        atomicAdd(&ctl->busy_ns[sm_phase], gtimer() - t_grab);
        atomicAdd(&ctl->grabs[sm_phase], 1u);
      }
      if (kind == 2 && p.max_leaders) atomicSub(&ctl->leaders, 1u);
    }
  }
}

__global__ void fill_ranges_kernel(uint8_t* base, const unsigned long long* offsets, const unsigned long long* bytes, int n,
                                   unsigned seed, unsigned lo, unsigned span, unsigned mode) {
  const int r = blockIdx.y;
  if (r >= n) return;
  for (unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < bytes[r];
       i += static_cast<unsigned long long>(gridDim.x) * blockDim.x) {
    const unsigned long long a = offsets[r] + i;
    if (mode == kFillF32Scales) {
      if (a & 3) continue;  // one word a float
      const unsigned long long w = a >> 2;
      unsigned h = static_cast<unsigned>(w * 2654435761ull) ^ static_cast<unsigned>(w >> 17) ^ seed;
      h ^= h >> 13; h *= 0x85ebca6bu; h ^= h >> 16;
      const float sv = 0.5f + (h % 1536) / 1024.0f;
      *reinterpret_cast<float*>(base + a) = sv;
      continue;
    }
    unsigned h = static_cast<unsigned>(a * 2654435761ull) ^ static_cast<unsigned>(a >> 17) ^ seed;
    h ^= h >> 13; h *= 0x85ebca6bu; h ^= h >> 16;
    if (mode == kFillE4m3)
      base[a] = static_cast<uint8_t>((h & 0x80) | ((1 + (h >> 8) % 13) << 3) | ((h >> 16) & 7));
    else if (mode == kFillBf16) {
      if (a & 1) continue;  // one element a pair of bytes: a small bf16 in [-0.25, 0.25)
      *reinterpret_cast<uint16_t*>(base + a) = float_to_bf16_bits(((h & 0xffff) / 65536.0f - 0.5f) * 0.5f);
    } else
      base[a] = static_cast<uint8_t>(lo + (h % span));
  }
}
__global__ void init_x_kernel(uint16_t* x, int n, unsigned seed) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned h = static_cast<unsigned>(i) * 2654435761u + seed; h ^= h >> 15; h *= 0x2c1b3c6du; h ^= h >> 12;
  x[i] = float_to_bf16_bits(((h & 0xffff) / 65536.0f - 0.5f) * 0.25f);
}

}  // namespace

int step_stream_capacity(unsigned smem_bytes) {
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return 0;
  int sms = 0;
  if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) return 0;
  if (smem_bytes > 48u * 1024u &&
      cudaFuncSetAttribute(step_stream_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)) != cudaSuccess)
    return 0;
  int per_sm = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, step_stream_kernel, kStepThreads, smem_bytes) != cudaSuccess) return 0;
  return per_sm * sms;
}

bool step_stream_has_mxfp4_consumer(int hidden, int inter) { return hidden == kConsumerH && inter == kConsumerI; }
bool step_stream_has_fp8_matrix(int k, int rows) {
  const int shapes[8][2] = {{5120, 1280}, {5120, 512}, {1280, 8192}, {4096, 2048}, {2048, 5120}, {1280, 4096}, {1536, 25600}, {15360, 5120}};
  for (const auto& s : shapes)
    if (s[0] == k && s[1] == rows) return true;
  return false;
}
unsigned step_stream_mxfp4_smem(int inter) { return step_stream_consumer_smem(inter, true); }
bool step_stream_has_router(int k) { return k == 5120; }
unsigned step_stream_consumer_smem(int inter, bool fp8, bool mhc, bool attention) {
  unsigned need = mhc ? kMhcUnitSmem + static_cast<unsigned>(kMhcFinishSmemFloats) * 4u : 0u;  // the hyper-connection unit's slice exchange and its finish's
  if (attention) {  // the flash body's tile, exchange and scratch; the finish's block
    const size_t flash = dgpp::dsa_flash::attn_flash_smem_bytes<kAttnLatent, kAttnLatent, true, kAttnSplitGroup>();
    need = std::max(need, static_cast<unsigned>(std::max(flash, sizeof(csa2_finish::FinishSmem))) + 256u);
  }
  if (inter == kConsumerI) need = static_cast<unsigned>(mxfp4_unit_smem_bytes<kConsumerI>());
  if (fp8) {  // the fp8 forms and the router unit (eight tokens by 5120 bf16 staged: 80 KB) fit the staged cap
    const unsigned split = static_cast<unsigned>(fp8_split_smem_bytes<kSplitMaxK, kSplitRowTiles>());
    const unsigned staged = static_cast<unsigned>(kFp8StagedSmemCap);  // the one-token-group projection form stages its slice
    need = std::max(need, std::max(split, staged));
  }
  return need;
}

cudaError_t launch_step_stream(const StepStreamParams& p, int workers, cudaStream_t stream) {
  if (workers < 1 || workers + 1 > step_stream_capacity(p.consumer_smem)) return cudaErrorInvalidConfiguration;
  step_stream_kernel<<<workers + 1, kStepThreads, p.consumer_smem, stream>>>(p);
  return cudaGetLastError();
}

void step_stream_fill_ranges(uint8_t* base, const unsigned long long* offsets, const unsigned long long* bytes, int n, unsigned seed,
                             unsigned lo, unsigned span, cudaStream_t stream, StepFillMode mode) {
  if (n <= 0) return;
  fill_ranges_kernel<<<dim3(64, static_cast<unsigned>(n)), 256, 0, stream>>>(base, offsets, bytes, n, seed, lo, span, static_cast<unsigned>(mode));
}
void step_stream_init_x(uint16_t* x, int n, unsigned seed, cudaStream_t stream) {
  init_x_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x, n, seed);
}

}  // namespace dgpp::step
