// The step engine's streamer (kernels/step/stream.hpp).
#include "kernels/step/stream.hpp"

#include "kernels/step/fp8_dense.cuh"
#include "kernels/step/mxfp4_expert.cuh"
#include "kernels/step/mxfp4_row.cuh"

namespace dgpp::step {

namespace {

constexpr unsigned kStateIdle = 0, kStateStream = 1, kStateEnd = 2;
// The MXFP4 consumer's shape compiled into this kernel: DeepSeek-V4.1 at
// world 4. Other shapes come with their family ports.
constexpr int kConsumerH = 5120, kConsumerI = 576;
constexpr int kConsumerFp8Block = 32;  // DeepSeek-V4.1's fp8 scale grid

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
__device__ __forceinline__ unsigned reduce_units(const StepStreamParams& p, const DevPool& pool) {
  if (pool.kind != kPoolFp8Matrix || pool.slabs <= 1) return 0;
  return (static_cast<unsigned>(pool.rows) * static_cast<unsigned>(p.ctl->rows) + kStepAccChunk - 1) / kStepAccChunk;
}
__device__ __forceinline__ unsigned pool_units(const StepStreamParams& p, int layer, const DevPool& pool) {
  const unsigned groups = pool.group_kind == kGroupRouted ? static_cast<unsigned>(p.ctl->run_count[layer]) : 1u;
  return groups * static_cast<unsigned>(pool.units_per_group) + reduce_units(p, pool);
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
__device__ __forceinline__ int unit_slot(const StepStreamParams& p, int layer, int P, unsigned u, unsigned* v) {
  const DevPhase& ph = p.phases[P];
  if (u < ph.fixed_units) { *v = u; return -1; }
  u -= ph.fixed_units;
  for (int i = 0; i < ph.n_pools; ++i) {
    if (i == ph.routed) continue;
    const unsigned n = pool_units(p, layer, p.pools[ph.pools[i]]);
    if (u < n) { *v = u; return i; }
    u -= n;
  }
  *v = u;
  return ph.routed;
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
__device__ bool locate_lead(const StepStreamParams& p, int layer, int P, unsigned u, const uint8_t** ptr, unsigned* bytes) {
  const DevPhase& ph = p.phases[P];
  const uint8_t* base = p.bases[layer];
  for (int i = 0; i < ph.seg_count; ++i)
    if (locate_segment(p, base, p.segments[ph.seg_begin + i], u, ptr, bytes)) return true;
  for (int i = 0; i < ph.lead_seg_count; ++i)
    if (locate_segment(p, base, p.segments[ph.lead_seg_begin + i], u, ptr, bytes)) return true;
  return false;
}
// Unit u of phase P in `layer`: its bytes in the layer's span.
__device__ bool locate(const StepStreamParams& p, int layer, int P, unsigned u, const uint8_t** ptr, unsigned* bytes) {
  const DevPhase& ph = p.phases[P];
  const uint8_t* base = p.bases[layer];
  if (u < ph.fixed_units) {
    for (int i = 0; i < ph.seg_count; ++i)
      if (locate_segment(p, base, p.segments[ph.seg_begin + i], u, ptr, bytes)) return true;
    return false;
  }
  if (ph.routed < 0) return false;
  const DevPool& pool = p.pools[ph.pools[ph.routed]];
  const unsigned v = u - ph.fixed_units;
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

// Streams `bytes` from `ptr` through the block: lane-contiguous 16-byte
// loads, four in flight a thread, folded into one word so the loads live.
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
      a = __ldcg(q + i); b = __ldcg(q + i + kStepThreads); c = __ldcg(q + i + 2 * kStepThreads); d = __ldcg(q + i + 3 * kStepThreads);
    }
    acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.y ^ c.z ^ c.w ^ d.x ^ d.y ^ d.z ^ d.w;
  }
  for (; i < n; i += kStepThreads) {
    const uint4 a = kEvictFirst ? __ldcs(q + i) : __ldcg(q + i);
    acc ^= a.x ^ a.y ^ a.z ^ a.w;
  }
  return acc;
}

// The synthetic router: `rows` rows pick `top_k` distinct groups each (a
// counter-based hash per pass, layer and row), the distinct picks sorted
// ascending into the layer's run list. Block-wide, the chain block's.
__device__ void route(const StepStreamParams& p, int layer, const DevPool& pool, int* keys, int* sorted) {
  const int rows = static_cast<int>(p.ctl->rows), top_k = static_cast<int>(p.ctl->top_k);
  const int picks = rows * top_k;
  if (threadIdx.x < static_cast<unsigned>(rows)) {
    unsigned h = p.ctl->seed * 2654435761u + 7919u * static_cast<unsigned>(layer) + 104729u * threadIdx.x;
    for (int i = 0; i < top_k; ++i) {
      int e;
      bool dup;
      do {
        h = h * 1664525u + 1013904223u;
        e = static_cast<int>((h >> 8) % static_cast<unsigned>(pool.groups));
        dup = false;
        for (int j = 0; j < i; ++j) dup = dup || keys[threadIdx.x * top_k + j] == e;
      } while (dup);
      keys[threadIdx.x * top_k + i] = e;
    }
  }
  __syncthreads();
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
__device__ void accumulate_chunk(const StepStreamParams& p, unsigned unit) {
  const int rows = static_cast<int>(p.ctl->rows), top_k = static_cast<int>(p.ctl->top_k), spt = top_k + 1, H = p.hidden;
  const int i0 = static_cast<int>(unit) * kStepAccChunk, i1 = min(i0 + kStepAccChunk, rows * H);
  const float w = 1.0f / static_cast<float>(top_k);
  for (int i = i0 + static_cast<int>(threadIdx.x); i < i1; i += kStepThreads) {
    const int t = i / H, c = i - t * H;
    float a = 0.f;
    for (int j = 0; j < top_k; ++j) a = __fmaf_rn(w, p.out[step_out_index(t * spt + j, c, H, spt)], a);
    a = __fmaf_rn(1.0f, p.out[step_out_index(t * spt + top_k, c, H, spt)], a);
    p.y[static_cast<size_t>(t) * H + c] = f32_to_bf16(a);
  }
}

// A slabbed projection's reduce unit: chunk `unit` of its [slots, rows]
// outputs as the sum of the slab partials in slab order, rounded once.
__device__ void reduce_unit(const StepStreamParams& p, const DevPool& pool, unsigned unit) {
  const int rows = static_cast<int>(p.ctl->rows), N = pool.rows;
  const int i0 = static_cast<int>(unit) * kStepAccChunk, i1 = min(i0 + kStepAccChunk, rows * N);
  const float* part = p.proj_part + pool.part_offset;
  uint16_t* y = p.proj + pool.out_offset;
  const size_t slab_stride = static_cast<size_t>(kStepMaxRows) * N;
  for (int i = i0 + static_cast<int>(threadIdx.x); i < i1; i += kStepThreads) {
    const int t = i / N, c = i - t * N;  // slot t is token t for a projection
    const size_t at = static_cast<size_t>(t) * N + c;
    float a = 0.f;
    for (int s = 0; s < pool.slabs; ++s) a += part[static_cast<size_t>(s) * slab_stride + at];
    y[at] = f32_to_bf16(a);
  }
}

// A consumer pool's unit: group g's 128-row block rb of the expert's pair or
// down matrix. `stage`: the group's activations are not yet in shared memory.
// The shared expert's pools have one group, every token's shared slot.
// The fp8 projection form at the shapes compiled in (DeepSeek-V4.1 at world 4).
__device__ bool fp8_matrix_dispatch(int k, int n, const uint8_t* w, const float* scales, const Mxfp4Group& g, int rb, const uint16_t* x,
                                    uint16_t* y, float* partial, uint8_t* smem, int w0, int w1, int unit_rows) {
#define DGPP_STEP_FP8_SHAPE(K, N)                                                                        \
  if (k == (K) && n == (N)) {                                                                            \
    fp8_matrix<(K), (N), kConsumerFp8Block>(w, scales, g, rb, x, y, partial, smem, w0, w1, unit_rows);  \
    return true;                                                                                         \
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
  if (pool.kind == kPoolFp8Matrix) {
    // Unit v: rb * slabs + slab for the compute units, then the reduce units.
    const int windows = pool.k / kMxfp4WindowCodes, per_slab = windows / pool.slabs;
    const int slab = rb % pool.slabs, block = rb / pool.slabs;
    const int w0 = slab * per_slab, w1 = w0 + per_slab;
    float* partial = pool.slabs > 1 ? p.proj_part + pool.part_offset + static_cast<size_t>(slab) * kStepMaxRows * pool.rows : nullptr;
    if (!fp8_matrix_dispatch(pool.k, pool.rows, base + pool.offsets[0], reinterpret_cast<const float*>(base + pool.offsets[1]), grp, block,
                             p.x, p.proj + pool.out_offset, partial, smem, w0, w1, pool.unit_rows) &&
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
          ctl->seed = d->seed + static_cast<unsigned>(d->seq);
          for (int P = 0; P < n_phases; ++P) { ctl->next[P] = 0; ctl->done[P] = 0; ctl->pf_units[P] = 0; ctl->idle_first[P] = 0; ctl->pool_done[P] = 0; }
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
              spin_us(ph.latency_us);
            }
            // The phase's bytes are consumed, the lead's among them: its cap frees up.
            atomicAdd(&ctl->consumed_pf_bytes, static_cast<unsigned long long>(ctl->pf_units[P]) * p.unit_bytes);
            ctl->stamp_end[P] = gtimer();
            if (ph.merged_next) {
              atomicAdd(&ctl->consumed_pf_bytes, static_cast<unsigned long long>(ctl->pf_units[P + 1]) * p.unit_bytes);
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
            if (pool_index >= 0) route(p, l, p.pools[pool_index], keys, sorted);
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
  __shared__ unsigned sm_kind, sm_unit, sm_phase, sm_count;
  __shared__ int sm_layer;
  for (;;) {
    if (threadIdx.x == 0) {
      sm_kind = 0;
      sm_count = 1;
      const unsigned w = ld_acq(&ctl->phase_word);
      const unsigned state = w & 3u;
      const int P = static_cast<int>((w >> 2) & 0xfffffu);
      if (state == kStateEnd || ld_acq(&ctl->poison)) {
        sm_kind = 3;
      } else {
        const int cur_layer = p.phases[P].layer;
        if (state == kStateStream) {
          const unsigned need = phase_units(p, cur_layer, P);
          const DevPhase& gph = p.phases[P];
          // The grab size: a consumer pool's at the counter's position, else one.
          // Every unit of the grab is classified on its own below, so a grab
          // that crosses a class boundary is still executed right.
          unsigned grab = 1;
          {
            const unsigned at = ld_acq(&ctl->next[P]);
            const unsigned own = own_units(p, cur_layer, P);
            int ip = P;
            unsigned iu = at;
            if (gph.merged_next && at >= own) { ip = P + 1; iu = at - own; }
            unsigned v = 0;
            const int slot = unit_slot(p, cur_layer, ip, iu, &v);
            const DevPhase& iph = p.phases[ip];
            if (slot >= 0 && p.pools[iph.pools[slot]].kind != kPoolBytes) grab = static_cast<unsigned>(p.pools[iph.pools[slot]].grab);
          }
          const unsigned u = atomicAdd(&ctl->next[P], grab);
          if (u < need) {
            sm_kind = 1; sm_unit = u; sm_phase = static_cast<unsigned>(P); sm_layer = cur_layer;
            sm_count = min(grab, need - u);
          } else if (ctl->idle_first[P] == 0) {
            ctl->idle_first[P] = gtimer();
          }
        }
        if (sm_kind == 0 && p.lead_cap_bytes) {
          long long ahead = static_cast<long long>(ctl->pf_bytes) - static_cast<long long>(ctl->consumed_pf_bytes);
          if (ahead < 0) ahead = 0;
          if (static_cast<unsigned long long>(ahead) + p.unit_bytes <= p.lead_cap_bytes) {
            bool slot = true;
            if (p.max_leaders) {
              if (atomicAdd(&ctl->leaders, 1u) >= p.max_leaders) { atomicSub(&ctl->leaders, 1u); slot = false; }
            }
            if (slot) {
              const unsigned pf = ld_acq(&ctl->pf_next);
              if (pf < p.lead_units) {
                const int P2 = phase_of_lead_unit(p, n_phases, pf);
                if (p.phases[P2].layer <= cur_layer + p.lookahead_layers) {
                  const unsigned got = atomicAdd(&ctl->pf_next, 1u);
                  if (got < p.lead_units) {
                    const int P3 = phase_of_lead_unit(p, n_phases, got);
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
        accumulate_chunk(p, xunit);
        continue;
      }
      unsigned xv = 0;
      const int xslot = kind == 1 ? unit_slot(p, sm_layer, static_cast<int>(xphase), xunit, &xv) : -1;
      const int pool_id = xslot >= 0 ? uph.pools[xslot] : -1;
      const DevPool* upool = pool_id >= 0 ? &p.pools[pool_id] : nullptr;
      if (upool && upool->kind != kPoolBytes) {
        const bool shared = upool->group_kind == kGroupShared;
        const unsigned compute_units = (upool->group_kind == kGroupRouted ? static_cast<unsigned>(p.ctl->run_count[sm_layer]) : 1u) *
                                       static_cast<unsigned>(upool->units_per_group);
        if (xv >= compute_units) {
          // A reduce unit: after the pool's compute units, which were handed out before it.
          if (threadIdx.x == 0) {
            const unsigned long long t_wait = gtimer();
            while (ld_acq(&ctl->pool_done[pool_id]) < compute_units) {
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
        const int g = static_cast<int>(xv / static_cast<unsigned>(upool->units_per_group));
        const int rb = static_cast<int>(xv % static_cast<unsigned>(upool->units_per_group));
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
          }
          __syncthreads();
        }
        consumer_unit(p, sm_layer, *upool, g, rb, smem, pool_id != last_pool || g != last_group);
        last_group = g;
        last_pool = pool_id;
        if (upool->kind == kPoolFp8Matrix && upool->slabs > 1) {
          __syncthreads();
          if (threadIdx.x == 0) { __threadfence(); atomicAdd(&ctl->pool_done[pool_id], 1u); }
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
        const bool found = kind == 2 ? locate_lead(p, sm_layer, static_cast<int>(xphase), xunit, &ptr, &bytes)
                                     : locate(p, sm_layer, static_cast<int>(xphase), xunit, &ptr, &bytes);
        if (found) {
          acc = kind == 2 ? stream_unit<false>(ptr, bytes) : stream_unit<true>(ptr, bytes);
        } else if (threadIdx.x == 0) {
          atomicCAS(&ctl->status, kStepOk, kStepBadUnit);
        }
        for (int o = 16; o; o >>= 1) acc ^= __shfl_xor_sync(0xffffffffu, acc, o);
        if ((threadIdx.x & 31) == 0) p.sink[(blockIdx.x % kStepMaxWorkers) * 8 + (threadIdx.x >> 5)] ^= acc;
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      if (kind == 1) { __threadfence(); atomicAdd(&ctl->done[sm_phase], sm_count); }
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
    else
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
unsigned step_stream_mxfp4_smem(int inter) { return inter == kConsumerI ? static_cast<unsigned>(mxfp4_unit_smem_bytes<kConsumerI>()) : 0u; }

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
