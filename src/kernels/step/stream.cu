// The step engine's streamer (kernels/step/stream.hpp).
#include "kernels/step/stream.hpp"

#include "kernels/step/fp8_dense.cuh"
#include "kernels/step/mhc.cuh"
#include "kernels/step/mxfp4_expert.cuh"
#include "kernels/step/mxfp4_row.cuh"

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
__device__ __forceinline__ unsigned compute_units_per_group(const StepStreamParams& p, const DevPool& pool) {
  if (projection_split(p, pool)) return (static_cast<unsigned>(pool.rows) + kSplitUnitRows - 1) / kSplitUnitRows;
  if (pool.kind == kPoolMhc) return static_cast<unsigned>(pool.units_per_group) * p.ctl->rows;  // a token's units for every token
  return static_cast<unsigned>(pool.units_per_group);
}
// A slabbed projection's reduce units: handed out after every pool's compute
// units and the fixed units (below), so the pool they wait on has only its
// stragglers left; handed out right after their own pool's compute units they
// spun on it, 58 of qkv's 211 units on DeepSeek-V4.1 (2026-10-30). A single
// worker reducing as the pool's last arriver was worse: the largest pool's
// reduce became the phase's tail.
__device__ __forceinline__ unsigned reduce_units(const StepStreamParams& p, const DevPool& pool) {
  if (pool.kind != kPoolFp8Matrix || pool.slabs <= 1 || projection_split(p, pool)) return 0;
  return (static_cast<unsigned>(pool.rows) * static_cast<unsigned>(p.ctl->rows) + kStepAccChunk - 1) / kStepAccChunk;
}
__device__ __forceinline__ unsigned pool_units(const StepStreamParams& p, int layer, const DevPool& pool) {
  const unsigned groups = pool.group_kind == kGroupRouted ? static_cast<unsigned>(p.ctl->run_count[layer]) : 1u;
  return groups * compute_units_per_group(p, pool) + reduce_units(p, pool);
}
__device__ __forceinline__ unsigned own_units(const StepStreamParams& p, int layer, int P) {
  const DevPhase& ph = p.phases[P];
  if (ph.accumulate) return (static_cast<unsigned>(p.ctl->rows) * static_cast<unsigned>(p.hidden) + kStepAccChunk - 1) / kStepAccChunk;
  unsigned n = ph.fixed_units;
  for (int i = 0; i < ph.n_pools; ++i) n += pool_units(p, layer, p.pools[ph.pools[i]]);
  return n;
}
// Where unit u of phase P sits: the fixed segments (-1), or a pool's slot
// in the phase's list, in the hand-out order: fixed, the pools that do not
// depend on the router in order (the longest units first), the routed pool
// last. `v` is the unit's index inside.
__device__ __forceinline__ unsigned pool_compute_units(const StepStreamParams& p, int layer, const DevPool& pool) {
  const unsigned groups = pool.group_kind == kGroupRouted ? static_cast<unsigned>(p.ctl->run_count[layer]) : 1u;
  return groups * compute_units_per_group(p, pool);
}
// (Handing out each class's units from the last to the first, so the staged
// units land in the tail where the lead fills the memory, measured within
// the noise on 2026-10-31: the order is the consumers'.)
__device__ __forceinline__ int unit_slot(const StepStreamParams& p, int layer, int P, unsigned u, unsigned* v) {
  const DevPhase& ph = p.phases[P];
  // The compute units: the pools that do not depend on the router, then the routed pool.
  for (int pass = 0; pass < 2; ++pass)
    for (int i = 0; i < ph.n_pools; ++i) {
      if ((pass == 0) == (i == ph.routed)) continue;
      const unsigned n = pool_compute_units(p, layer, p.pools[ph.pools[i]]);
      if (u < n) { *v = u; return i; }
      u -= n;
    }
  if (u < ph.fixed_units) { *v = u; return -1; }  // the fixed byte units: the shortest, they fill the tail
  u -= ph.fixed_units;
  // The reduce units last, after every pool's compute units: v counts past the pool's compute units.
  for (int pass = 0; pass < 2; ++pass)
    for (int i = 0; i < ph.n_pools; ++i) {
      if ((pass == 0) == (i == ph.routed)) continue;
      const DevPool& pool = p.pools[ph.pools[i]];
      const unsigned n = reduce_units(p, pool);
      if (u < n) { *v = pool_compute_units(p, layer, pool) + u; return i; }
      u -= n;
    }
  *v = u;
  return -1;
}
// The units a published phase hands out: its own, and the merged next phase's after them.
__device__ __forceinline__ unsigned phase_units(const StepStreamParams& p, int layer, int P) {
  unsigned n = own_units(p, layer, P);
  if (p.phases[P].merged_next) n += own_units(p, layer, P + 1);
  return n;
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
__device__ int phase_of_lead_unit(const StepStreamParams& p, int n_phases, unsigned g) {
  int lo = 0, hi = n_phases - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (p.phases[mid].lead_begin <= g) lo = mid; else hi = mid - 1;
  }
  return lo;
}
// The same over a shared-memory copy of the phases' lead_begin: the workers'
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
__device__ void route_finish(const StepStreamParams& p, int layer, int* keys, int* sorted) {
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
      if (sorted[o] != prev) { p.ctl->run_list[layer][n++] = sorted[o]; prev = sorted[o]; }
    p.ctl->run_count[layer] = n;
  }
  __syncthreads();
  // Group g: the slots of run_list[g], slot = row * (top_k + 1) + pick, in row order.
  if (threadIdx.x < static_cast<unsigned>(p.ctl->run_count[layer])) {
    const int g = static_cast<int>(threadIdx.x), expert = p.ctl->run_list[layer][g];
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
  if (fused) mhc_update_chunk(p.post, p.comb, p.y, p.streams[1], p.streams[0], H, i0, i1);  // the same thread owns the same elements
}

// A hyper-connection site's unit: kMhcUnitCoeffs logits of one token (unit
// rb = token * units + group), the arrival that completes the token's 24
// runs its finish: pre, post, the Sinkhorn comb, the collapse and the normed
// row into x. Site s reads streams[s] and pre[s], writes pre[s ^ 1].
__device__ void mhc_unit(const StepStreamParams& p, int layer, const DevPool& pool, const uint8_t* base, int rb, uint8_t* smem) {
  __shared__ unsigned sm_last;
  constexpr int kUnits = kMhcCoeffs / kMhcUnitCoeffs;  // a token's units
  const int H = p.hidden, s = pool.site;
  const int t = rb / kUnits, c0 = (rb % kUnits) * kMhcUnitCoeffs;
  const uint16_t* streams = p.streams[s] + static_cast<size_t>(t) * kMhcN * H;
  const uint16_t* fn = reinterpret_cast<const uint16_t*>(base + pool.offsets[0]) + static_cast<size_t>(c0) * kMhcN * H;
  float* f = reinterpret_cast<float*>(smem);  // [0, 8) the dot's scratch, [32, 56) the logits row, [64, 92) the finish's scratch
  float lg[kMhcUnitCoeffs];
  mhc_dot_n<kMhcUnitCoeffs>(streams, fn, H, p.mhc_norm_eps, f, lg);
  if (threadIdx.x == 0) {
#pragma unroll
    for (int c = 0; c < kMhcUnitCoeffs; ++c) p.logits[t * kMhcCoeffs + c0 + c] = lg[c];
    __threadfence();
    const unsigned before = atomicAdd(&p.ctl->mhc_done[layer][s][t], static_cast<unsigned>(kMhcUnitCoeffs));
    sm_last = before + kMhcUnitCoeffs == static_cast<unsigned>(kMhcCoeffs) ? 1u : 0u;
    if (sm_last) p.ctl->mhc_done[layer][s][t] = 0;  // every logit of the token arrived: ready for the next step
  }
  __syncthreads();
  if (!sm_last) return;
  __threadfence();
  if (threadIdx.x < static_cast<unsigned>(kMhcCoeffs)) f[32 + threadIdx.x] = __ldcg(p.logits + t * kMhcCoeffs + threadIdx.x);
  __syncthreads();
  MhcSite site;
  site.base = reinterpret_cast<const float*>(base + pool.offsets[1]);
  site.scale = reinterpret_cast<const float*>(base + pool.offsets[2]);
  site.ln = reinterpret_cast<const uint16_t*>(base + pool.offsets[3]);
  site.hc_eps = p.hc_eps;
  site.sinkhorn_iters = p.sinkhorn_iters;
  site.ln_eps = p.mhc_norm_eps;
  mhc_finish(streams, f + 32, site, H, p.pre[s] + t * kMhcN, p.pre[s ^ 1] + t * kMhcN, p.post + t * kMhcN, p.comb + t * kMhcN * kMhcN,
             p.x + static_cast<size_t>(t) * p.x_stride, f + 64);
}

// A slabbed projection's reduce unit: chunk `unit` of its [slots, rows]
// outputs as the sum of the slab partials in slab order, rounded once.
// With `update_site`, the attention site's stream update follows on the
// chunk: the pool's y (the output projection, [token][hidden]) is h, and
// streams[0] mix into streams[1] under the site's post and comb.
__device__ void reduce_unit(const StepStreamParams& p, const DevPool& pool, unsigned unit) {
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
  if (pool.update_site) mhc_update_chunk(p.post, p.comb, y, p.streams[0], p.streams[1], N, i0, i1);
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

// ---- the router -----------------------------------------------------------
// The production router's dot (kernels/glm_moe.cu router_dot), op for op: one
// warp per expert row, each lane its strided 16-byte vectors of the row and
// of the token's activations, eight fmas a vector in the file's order, then
// the xor tree. Here a warp carries up to eight tokens' accumulators through
// one read of its row, with the activations staged in shared memory; each
// token's chain is the production's, so the scores are bitwise its.
constexpr int kRouterUnitRows = 16;  // two expert rows a warp, sharing each staged activation vector: 24 units in one wave
constexpr int kRouterTokens = 8;  // tokens a pass stages
__device__ __forceinline__ float router_sqrt_softplus(float x) {
  const float sp = x > 20.f ? x : log1pf(expf(x));
  return sqrtf(sp);
}
template <int kK>
__device__ void router_unit(const uint8_t* __restrict__ w_base, const DevPool& pool, int rb, const Mxfp4Group& g, const uint16_t* __restrict__ x,
                            float* __restrict__ scores, float* __restrict__ biased, int n_experts, int mode, uint8_t* smem) {
  static_assert(kK % 256 == 0, "the router's inner width");
  const uint16_t* w = reinterpret_cast<const uint16_t*>(w_base + pool.offsets[0]);
  const float* bias = reinterpret_cast<const float*>(w_base + pool.offsets[1]);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  uint16_t* s_x = reinterpret_cast<uint16_t*>(smem);
  const size_t xs = g.x_stride ? static_cast<size_t>(g.x_stride) : static_cast<size_t>(kK);
  for (int t0 = 0; t0 < g.count; t0 += kRouterTokens) {
    const int nt = min(kRouterTokens, g.count - t0);
    __syncthreads();  // the previous pass's readers are done with the staged rows
    for (int q = threadIdx.x; q < kRouterTokens * (kK / 8); q += kStepThreads) {  // rows past nt stage row 0 again: finite operands for the idle accumulators
      const int t = q / (kK / 8), c16 = q % (kK / 8);
      cp_async_16(s_x + t * kK + c16 * 8, x + static_cast<size_t>(g.tokens[t0 + (t < nt ? t : 0)]) * xs + c16 * 8);
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();
    // Both of the warp's rows against every staged token through one read of each activation
    // vector; each (row, token) chain is the production's lane order, so the scores are bitwise.
    const int e0 = rb * kRouterUnitRows + warp, e1 = e0 + (kStepThreads / 32);
    const bool live1 = e1 < n_experts;
    if (e0 < n_experts) {
      const uint4* wv0 = reinterpret_cast<const uint4*>(w + static_cast<size_t>(e0) * kK);
      const uint4* wv1 = reinterpret_cast<const uint4*>(w + static_cast<size_t>(live1 ? e1 : e0) * kK);
      float dot0[kRouterTokens], dot1[kRouterTokens];
#pragma unroll
      for (int t = 0; t < kRouterTokens; ++t) { dot0[t] = 0.f; dot1[t] = 0.f; }
#pragma unroll 2
      for (int v = lane; v < kK / 8; v += 32) {
        const uint4 wq0 = __ldcs(wv0 + v), wq1 = __ldcs(wv1 + v);
        const uint32_t w0[4] = {wq0.x, wq0.y, wq0.z, wq0.w}, w1[4] = {wq1.x, wq1.y, wq1.z, wq1.w};
#pragma unroll
        for (int t = 0; t < kRouterTokens; ++t) {
          const uint4 xq = *reinterpret_cast<const uint4*>(s_x + t * kK + v * 8);
          const uint32_t xw[4] = {xq.x, xq.y, xq.z, xq.w};
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const float xl = bf16_bits_to_float(static_cast<uint16_t>(xw[i] & 0xFFFFu)), xh = bf16_bits_to_float(static_cast<uint16_t>(xw[i] >> 16));
            dot0[t] = __fmaf_rn(xl, bf16_bits_to_float(static_cast<uint16_t>(w0[i] & 0xFFFFu)), dot0[t]);
            dot0[t] = __fmaf_rn(xh, bf16_bits_to_float(static_cast<uint16_t>(w0[i] >> 16)), dot0[t]);
            dot1[t] = __fmaf_rn(xl, bf16_bits_to_float(static_cast<uint16_t>(w1[i] & 0xFFFFu)), dot1[t]);
            dot1[t] = __fmaf_rn(xh, bf16_bits_to_float(static_cast<uint16_t>(w1[i] >> 16)), dot1[t]);
          }
        }
      }
#pragma unroll
      for (int t = 0; t < kRouterTokens; ++t) {
        float d0 = dot0[t], d1 = dot1[t];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) { d0 += __shfl_xor_sync(0xFFFFFFFFu, d0, off); d1 += __shfl_xor_sync(0xFFFFFFFFu, d1, off); }
        if (lane == 0 && t < nt) {
          const int tok = g.tokens[t0 + t];
          const float s0 = mode == 2 ? router_sqrt_softplus(d0) : 1.0f / (1.0f + expf(-d0));
          scores[static_cast<size_t>(tok) * n_experts + e0] = s0;
          biased[static_cast<size_t>(tok) * n_experts + e0] = s0 + bias[e0];
          if (live1) {
            const float s1 = mode == 2 ? router_sqrt_softplus(d1) : 1.0f / (1.0f + expf(-d1));
            scores[static_cast<size_t>(tok) * n_experts + e1] = s1;
            biased[static_cast<size_t>(tok) * n_experts + e1] = s1 + bias[e1];
          }
        }
      }
    }
  }
}
// The production router's select (kernels/glm_moe.cu router_select_warp), one
// warp per token: top_k rounds of argmax over the biased scores, strict > and
// the lower id on ties, the ids sorted ascending, the weights normalized with
// per-element division and scaled. keys[t * top_k + i] takes the ids, the
// control block's slot_weight the weights, 1 for the shared slot.
__device__ __forceinline__ void router_argmax_lowest_id(float& v, int& id) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const float ov = __shfl_xor_sync(0xFFFFFFFFu, v, off);
    const int oid = __shfl_xor_sync(0xFFFFFFFFu, id, off);
    if (ov > v || (ov == v && oid < id)) { v = ov; id = oid; }
  }
}
__device__ void route_select(const StepStreamParams& p, int layer, int* keys, uint8_t* smem) {
  const int rows = static_cast<int>(p.ctl->rows), top_k = static_cast<int>(p.ctl->top_k), spt = top_k + 1, E = p.n_experts;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  float* s_scores = reinterpret_cast<float*>(smem);  // [8][E] scores, [8][E] biased, then each warp's sel [16] and wsel [16]
  for (int t0 = 0; t0 < rows; t0 += 8) {
    __syncthreads();
    const int t = t0 + warp;
    if (t < rows) {
      float* sc = s_scores + warp * E;
      float* bs = s_scores + (8 + warp) * E;
      for (int e = lane; e < E; e += 32) {
        sc[e] = __ldcg(p.scores + static_cast<size_t>(t) * E + e);
        bs[e] = __ldcg(p.biased + static_cast<size_t>(t) * E + e);
      }
      __syncwarp();
      int* sel = reinterpret_cast<int*>(smem + static_cast<size_t>(16) * E * 4) + warp * 16;        // [8][16] after the score rows
      float* wsel = reinterpret_cast<float*>(smem + static_cast<size_t>(16) * E * 4 + 8 * 16 * 4) + warp * 16;
      for (int r = 0; r < top_k; ++r) {
        int best = INT_MAX;
        float bv = -INFINITY;
        for (int e = lane; e < E; e += 32)
          if (bs[e] > bv) { bv = bs[e]; best = e; }
        router_argmax_lowest_id(bv, best);
        sel[r] = best;
        wsel[r] = sc[best];
        __syncwarp();
        if (lane == 0) bs[best] = -INFINITY;
        __syncwarp();
      }
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
  if (pool.kind == kPoolMhc) {
    mhc_unit(p, layer, pool, base, rb, smem);
    return;
  }
  if (pool.kind == kPoolRouter) {
    if (pool.k == 5120) router_unit<5120>(base, pool, rb, grp, p.x, p.scores, p.biased, p.n_experts, p.router_mode, smem);
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
    if (!fp8_matrix_dispatch(pool.k, pool.rows, base + pool.offsets[0], reinterpret_cast<const float*>(base + pool.offsets[1]), grp, block,
                             p.x, p.proj + pool.out_offset, partial, smem, w0, w1, pool.unit_rows, split, stage) &&
        threadIdx.x == 0)
      atomicCAS(&p.ctl->status, kStepOk, kStepBadUnit);
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
          ctl->top_k = d->top_k;
          // The token group's slots are the step's rows from the start: the first layer's router reads them before any route ran.
          for (int i = 0; i < 32; ++i) ctl->token_slots[i] = i;
          ctl->seed = d->seed + static_cast<unsigned>(d->seq);
          ctl->route_pool = p.route_pool;
          for (int P = 0; P < n_phases; ++P) {
            ctl->next[P] = 0; ctl->done[P] = 0; ctl->pf_units[P] = 0; ctl->idle_first[P] = 0;
            for (int L = 0; L < kStepMaxLayers; ++L) ctl->pool_done[L][P] = 0;
            ctl->busy_ns[P] = 0; ctl->grabs[P] = 0; ctl->handout_ns[P] = 0; ctl->lead_in_bytes[P] = 0; ctl->pf_credit[P] = 0; ctl->pf_staged[P] = 0;
            ctl->wait_ns[P] = 0;
          }
          ctl->pf_next = 0;
          ctl->pf_bytes = 0;
          ctl->consumed_pf_bytes = 0;
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
      // ---- the layers
      for (int l = 0; l < p.n_layers && !sm_fail; ++l) {
        const DevLayer& dl = p.layers[l];
        for (int P = dl.phase_begin; P < dl.phase_begin + dl.phase_count; ++P) {
          const DevPhase& ph = p.phases[P];
          if (t0) {
            const unsigned long long t_begin = gtimer();
            ctl->stamp_begin[P] = t_begin;
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
            // The lead never takes the units of the phase that is running: a
            // stream phase's workers read them now, a latency phase's chain does.
            atomicMax(&ctl->pf_next, ph.lead_begin + ph.lead_units);
            if (ph.merged_next) atomicMax(&ctl->pf_next, p.phases[P + 1].lead_begin + p.phases[P + 1].lead_units);
            if (t0) {
              const long long ahead = static_cast<long long>(ctl->pf_bytes) - static_cast<long long>(ctl->consumed_pf_bytes);
              ctl->ahead_begin[P] = ahead > 0 ? static_cast<unsigned long long>(ahead) : 0ull;
              const unsigned pf = ld_acq(&ctl->pf_next);
              ctl->cursor_begin[P] = pf < p.lead_units ? static_cast<unsigned>(phase_of_lead_unit(p, n_phases, pf)) : static_cast<unsigned>(n_phases);
              ctl->look_blocked[P] = 0;
            }
            if (!sm_fail && ph.kind == 1) {
              st_rel(&ctl->phase_word, word(step, P, kStateStream));
              const unsigned need = phase_units(p, l, P);
              while (ld_acq(&ctl->done[P]) < need) {
                if (p.deadline_ns && gtimer() - t_begin > p.deadline_ns) {
                  atomicCAS(&ctl->status, kStepOk, kStepDeadline | (static_cast<unsigned>(P) << 8));
                  sm_fail = 1;
                  break;
                }
                if (ld_acq(&ctl->poison)) { atomicCAS(&ctl->status, kStepOk, kStepPoisoned); sm_fail = 1; break; }
              }
              st_rel(&ctl->phase_word, word(step, P, kStateIdle));
            } else if (!sm_fail) {
              // A latency phase is published too, idle, so the workers' lead sees the phase
              // the chain is in: its idle budget and the next stream phase's exemption.
              st_rel(&ctl->phase_word, word(step, P, kStateIdle));
              spin_us(ph.latency_us);
            }
            // The phase's bytes are consumed, the lead's among them: whatever the workers have
            // not credited unit by unit frees up now.
            atomicAdd(&ctl->consumed_pf_bytes, ctl->pf_staged[P] - ctl->pf_credit[P]);
            ctl->stamp_end[P] = gtimer();
            if (ph.merged_next) {
              atomicAdd(&ctl->consumed_pf_bytes, ctl->pf_staged[P + 1] - ctl->pf_credit[P + 1]);
              ctl->stamp_begin[P + 1] = ctl->stamp_begin[P];
              ctl->stamp_end[P + 1] = ctl->stamp_end[P];
            }
          }
          __syncthreads();
          if (sm_fail) break;
          if (ph.resolves == static_cast<uint8_t>(Dependency::Router)) {
            // The first pool of the layer's later phases says how many groups there are.
            int pool_index = -1;
            for (int Q = P + 1; Q < dl.phase_begin + dl.phase_count && pool_index < 0; ++Q)
              if (p.phases[Q].routed >= 0) pool_index = p.phases[Q].pools[p.phases[Q].routed];
            if (pool_index >= 0) {
              // The phase's own router pool, when it has one: the select over its scores; else the synthetic picks.
              bool real = false;
              for (int i = 0; i < ph.n_pools; ++i) real = real || p.pools[ph.pools[i]].kind == kPoolRouter;
              const unsigned long long t_sel = gtimer();
              if (real) {
                if (t0) __threadfence();
                route_select(p, l, keys, chain_smem);
              } else {
                route_synthetic_picks(p, l, p.pools[pool_index], keys);
              }
              const unsigned long long t_fin = gtimer();
              route_finish(p, l, keys, sorted);
              if (t0) { ctl->route_select_ns[l] = t_fin - t_sel; ctl->route_finish_ns[l] = gtimer() - t_fin; }
            }
            if (t0) __threadfence();
          }
          const float fold_us = ph.merged_next ? p.phases[P + 1].fold_after_us : ph.fold_after_us;
          if (t0 && fold_us > 0.f) {
            spin_us(fold_us);
            ctl->stamp_end[P] = gtimer();
            if (ph.merged_next) ctl->stamp_end[P + 1] = ctl->stamp_end[P];
          }
          __syncthreads();
          if (ph.merged_next) ++P;  // the merged phase ran inside this one
        }
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
  __shared__ unsigned sm_kind, sm_unit, sm_phase, sm_count, sm_word, sm_stop, sm_published, sm_look_blocked;
  __shared__ unsigned sm_lead_begin[kStepMaxPhases];  // the lead sequence's phase boundaries, for the lead's hand-out
  __shared__ unsigned long long sm_grab_bytes;
  __shared__ unsigned sm_cache_word, sm_need, sm_nslots;
  __shared__ unsigned sm_slot_end[2 * (kMaxPhasePools + 2)], sm_slot_grab[2 * (kMaxPhasePools + 2)];
  __shared__ int sm_layer;
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
          // The phase's hand-out table, built once a phase a worker: the unit count and, in the
          // hand-out order (the pools that do not depend on the router, the routed pool, the fixed
          // units, then the merged next phase's the same way), each class's end and grab size. A
          // grab then costs the counter's load and the atomic, where recomputing the table's
          // dependent loads under a saturated memory cost 3.5 to 4.5 us a grab (2026-10-30).
          const unsigned at = ld_acq(&ctl->next[P]);
          if (w != sm_cache_word) {
            unsigned n = 0, end = 0;
            for (int ip = P; ip <= (p.phases[P].merged_next ? P + 1 : P); ++ip) {
              const DevPhase& iph = p.phases[ip];
              if (iph.accumulate) {  // the ordered accumulation's chunks are the phase's only units
                end += own_units(p, cur_layer, ip);
                sm_slot_end[n] = end;
                sm_slot_grab[n] = 1u;
                ++n;
                continue;
              }
              for (int pass = 0; pass < 2; ++pass)
                for (int i = 0; i < iph.n_pools; ++i) {
                  if ((pass == 0) == (i == iph.routed)) continue;  // the routed pool in the second pass
                  const DevPool& gp = p.pools[iph.pools[i]];
                  end += pool_compute_units(p, cur_layer, gp);
                  sm_slot_end[n] = end;
                  sm_slot_grab[n] = gp.kind == kPoolBytes ? 1u : (projection_split(p, gp) ? 4u : static_cast<unsigned>(gp.grab));
                  ++n;
                }
              end += iph.fixed_units;
              sm_slot_end[n] = end;
              sm_slot_grab[n] = 1u;
              ++n;
              unsigned reduce = 0;  // every pool's reduce units, one class, grabbed singly
              for (int i = 0; i < iph.n_pools; ++i) reduce += reduce_units(p, p.pools[iph.pools[i]]);
              if (reduce) {
                end += reduce;
                sm_slot_end[n] = end;
                sm_slot_grab[n] = 1u;
                ++n;
              }
            }
            sm_need = end;
            sm_nslots = n;
            sm_cache_word = w;
          }
          const unsigned need = sm_need;
          // The grab size: the class's at the counter's position. Every unit of the grab is
          // classified on its own below, so a grab that crosses a class boundary is still
          // executed right. Consecutive units share their staging while the phase has units for
          // every worker; the tail takes them singly.
          unsigned grab = 1;
          for (unsigned i = 0; i < sm_nslots; ++i)
            if (at < sm_slot_end[i]) { grab = sm_slot_grab[i]; break; }
          if (grab > 1) {  // the tail shrinks the grab gradually: at most a worker's share of what remains
            const unsigned remaining = need > at ? need - at : 0u;
            const unsigned share = remaining / (gridDim.x - 1);
            grab = share < 1u ? 1u : (share < grab ? share : grab);
          }
          // (Workers leading inside the routed expert phase in place of taking its units were
          // measured 2026-10-31 and lost four to five points: that phase is memory-saturated,
          // so the staging added to it one to one and bought back a quarter of itself in qkv.)
          const unsigned u = atomicAdd(&ctl->next[P], grab);
          if (u < need) {
            sm_kind = 1; sm_unit = u; sm_phase = static_cast<unsigned>(P); sm_layer = cur_layer;
            sm_count = min(grab, need - u);
            atomicAdd(&ctl->handout_ns[P], gtimer() - t_top);
          } else if (ctl->idle_first[P] == 0) {
            ctl->idle_first[P] = gtimer();
          }
        }
        if (sm_kind == 0 && p.lead_cap_bytes) {
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
              const bool merged = qh.merged_next != 0;
              idle_us += merged ? p.phases[q + 1].fold_after_us : qh.fold_after_us;
              if (merged) ++q;
            }
            cap += static_cast<unsigned long long>(idle_us * p.lead_bytes_per_us);
          }
          if (static_cast<unsigned long long>(ahead) + p.unit_bytes <= cap) {
            bool slot = true;
            if (p.max_leaders) {
              if (atomicAdd(&ctl->leaders, 1u) >= p.max_leaders) { atomicSub(&ctl->leaders, 1u); slot = false; }
            }
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
                    sm_kind = 2; sm_unit = got - p.phases[P3].lead_begin; sm_phase = static_cast<unsigned>(P3); sm_layer = p.phases[P3].layer;
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
    // Each unit of the grab on its own: a unit past a merged phase's own
    // units belongs to the next phase, and inside a phase it is a fixed byte
    // unit, a routed consumer unit or a shared-expert unit.
    int last_group = -1, last_pool = -1;
    for (unsigned q = 0; q < sm_count; ++q) {
      const unsigned gu = sm_unit + q;
      unsigned xphase = sm_phase, xunit = gu;
      if (kind == 1 && p.phases[sm_phase].merged_next) {
        const unsigned own = own_units(p, sm_layer, static_cast<int>(sm_phase));
        if (gu >= own) { xphase = sm_phase + 1; xunit = gu - own; }
      }
      const DevPhase& uph = p.phases[xphase];
      if (q) __syncthreads();  // the previous unit's readers of the shared memory are done
      if (kind == 1 && uph.accumulate) {
        accumulate_chunk(p, sm_layer, xunit, uph.update != 0);
        continue;
      }
      unsigned xv = 0;
      const int xslot = kind == 1 ? unit_slot(p, sm_layer, static_cast<int>(xphase), xunit, &xv) : -1;
      const int pool_id = xslot >= 0 ? uph.pools[xslot] : -1;
      const DevPool* upool = pool_id >= 0 ? &p.pools[pool_id] : nullptr;
      if (upool && upool->kind != kPoolBytes) {
        const bool shared = upool->group_kind == kGroupShared;
        const unsigned compute_units = (upool->group_kind == kGroupRouted ? static_cast<unsigned>(p.ctl->run_count[sm_layer]) : 1u) *
                                       compute_units_per_group(p, *upool);
        if (xv >= compute_units) {
          // A reduce unit: after the pool's compute units, which were handed out before it.
          if (threadIdx.x == 0) {
            const unsigned long long t_wait = gtimer();
            while (ld_acq(&ctl->pool_done[sm_layer][pool_id]) < compute_units) {
              if (p.deadline_ns && gtimer() - t_wait > p.deadline_ns) { atomicCAS(&ctl->status, kStepOk, kStepDeadline | (xphase << 8)); break; }
              if (ld_acq(&ctl->poison)) break;
              __nanosleep(100);
            }
          }
          __syncthreads();
          reduce_unit(p, *upool, xv - compute_units);
          last_group = -1;
          last_pool = -1;
          continue;
        }
        const unsigned upg = compute_units_per_group(p, *upool);
        const int g = static_cast<int>(xv / upg);
        const int rb = static_cast<int>(xv % upg);
        if (pool_is_down(upool->kind) && upool->partner >= 0) {
          // The group's gate and up units must be complete: they were handed out earlier, so the wait is short.
          if (threadIdx.x == 0) {
            const unsigned gate_units = static_cast<unsigned>(p.pools[upool->partner].units_per_group);
            const unsigned* done = shared ? &ctl->shared_gate_done[sm_layer] : &ctl->gate_done[sm_layer][g];
            const unsigned long long t_wait = gtimer();
            while (ld_acq(done) < gate_units) {
              if (p.deadline_ns && gtimer() - t_wait > p.deadline_ns) { atomicCAS(&ctl->status, kStepOk, kStepDeadline | (xphase << 8)); break; }
              if (ld_acq(&ctl->poison)) break;
              __nanosleep(100);
            }
            atomicAdd(&ctl->wait_ns[sm_phase], gtimer() - t_wait);
          }
          __syncthreads();
        }
        consumer_unit(p, sm_layer, *upool, g, rb, smem, pool_id != last_pool || g != last_group);
        last_group = g;
        last_pool = pool_id;
        if (threadIdx.x == 0) sm_grab_bytes += upool->bytes / upg;
        if (upool->kind == kPoolFp8Matrix && upool->slabs > 1 && !projection_split(p, *upool)) {
          __syncthreads();
          if (threadIdx.x == 0) { __threadfence(); atomicAdd(&ctl->pool_done[sm_layer][pool_id], 1u); }
        }
        if (pool_is_gate_up(upool->kind)) {
          __syncthreads();
          if (threadIdx.x == 0) {
            __threadfence();
            atomicAdd(shared ? &ctl->shared_gate_done[sm_layer] : &ctl->gate_done[sm_layer][g], 1u);
          }
        }
      } else {
        last_group = -1;
        last_pool = -1;
        const uint8_t* ptr = nullptr;
        unsigned bytes = 0;
        unsigned acc = 0;
        const bool found = kind == 2  ? locate_lead(p, sm_layer, static_cast<int>(xphase), xunit, &ptr, &bytes)
                           : upool    ? locate_pool_bytes(p, sm_layer, *upool, xv, &ptr, &bytes)
                                      : locate_fixed(p, sm_layer, static_cast<int>(xphase), xv, &ptr, &bytes);
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
                sm_stop = (now != sm_word && (now & 3u) == kStateStream) ? 1u : 0u;  // a stream phase with units wants the worker
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
        __threadfence();
        atomicAdd(&ctl->done[sm_phase], sm_count);
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
unsigned step_stream_consumer_smem(int inter, bool fp8, bool mhc) {
  unsigned need = mhc ? 512u : 0u;  // the hyper-connection unit's scratch: 92 floats
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
