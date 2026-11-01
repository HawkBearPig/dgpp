#pragma once
// The step engine's streamer (MOONSHOT-1.md sections 2.1 to 2.6, in the
// form the phase-1 probes measured): one persistent kernel whose block 0 is
// the chain, driving a layer's phases in order and spinning the modeled
// latencies, and whose other blocks are workers that grab units of the
// released ranges by atomic counter and stream them at the ceiling with
// evict-first loads, four 16-byte loads in flight a lane. Idle workers run
// the lead: the fixed units of the phases ahead, within a lookahead and an
// L2 cap, with normal loads so the lines stay for the phase that consumes
// them. A governor bounds how many workers lead at once. Every spin has a
// deadline and exits through one path that publishes a status, and a host
// poison word ends the kernel from outside.
//
// Phase 2 consumes synthetically: a unit's bytes fold into one word a warp.
// The consumers of phase 3 replace stream_unit with their forms.
#include <cstdint>

#include <cuda_runtime.h>

#include "net/bus_kernel.hpp"
#include "step/device_plan.hpp"
#include "step/protocol.hpp"

namespace dgpp::step {

constexpr int kStepMaxPhases = 1024;
constexpr int kStepMaxClasses = 24;  // unit classes of one published phase group (its phases' pools, reduces and fixed units)
constexpr int kStepMaxLayers = 64;
constexpr int kStepMaxGroups = 512;  // run-list capacity a layer
constexpr int kStepMaxWorkers = 128;
constexpr int kStepThreads = 256;

enum StepStatus : unsigned {
  kStepOk = 0,
  kStepDeadline = 1,  // a spin ran past its deadline; the phase is in bits 8 and up
  kStepPoisoned = 2,  // the host set the poison word
  kStepBadUnit = 3,   // a worker was handed a unit outside its phase
};

// The control block the chain and the workers share (managed memory).
struct alignas(128) StepCtl {
  unsigned phase_word;  // (pass << 22) | (phase << 2) | state
  unsigned pad0[31];
  unsigned status;
  unsigned poison;
  unsigned pad1[30];
  unsigned pf_next;   // the lead's cursor over the global fixed-unit sequence
  unsigned leaders;   // workers streaming the lead now
  unsigned stage_active;  // workers streaming a merged phase's staging units now
  unsigned pad2[29];
  unsigned long long pf_bytes;           // bytes the lead has read
  unsigned long long consumed_pf_bytes;  // of them, the bytes of phases now complete
  unsigned pad3[28];
  // The hand-out counters of a published phase's unit classes (kernels/step/stream.cu: the
  // group's classes in order, each with its own counter and gate), and the group's completed units.
  unsigned cls_next[kStepMaxPhases][kStepMaxClasses];
  uint4 cls_table[kStepMaxPhases][kStepMaxClasses];  // the group's classes, built by the chain (kernels/step/stream.cu build_class_table)
  unsigned cls_n[kStepMaxPhases];
  unsigned done[kStepMaxPhases];
  unsigned pf_units[kStepMaxPhases];
  int run_count[kStepMaxLayers];
  int run_list[kStepMaxLayers][kStepMaxGroups];
  // The step's slots by group (the router's output): group g holds the
  // slots of one expert of the run list, at most 32, slot i's token row.
  int32_t group_slots[kStepMaxLayers][kStepMaxGroups][32];
  int32_t group_tokens[kStepMaxLayers][kStepMaxGroups][32];
  int32_t group_count[kStepMaxLayers][kStepMaxGroups];
  unsigned gate_done[kStepMaxLayers][kStepMaxGroups];  // gate and up units of the group completed this step
  // The shared expert's group: every token's shared slot (slot = t * (top_k + 1) + top_k).
  int32_t shared_slots[kStepMaxLayers][32];
  int32_t shared_tokens[kStepMaxLayers][32];
  unsigned shared_gate_done[kStepMaxLayers];
  int32_t token_slots[32];  // the token group: slot i = token i
  // The accumulation's weight of every slot of the running step, by layer: a token's routed
  // picks in ascending expert order (the router's normalized, scaled weights, or 1 / top_k
  // from the synthetic router), then 1 for its shared slot. slot = row * (top_k + 1) + pick.
  float slot_weight[kStepMaxLayers][32 * 9];
  // A hyper-connection site's (coefficient, token) dots arrived, by layer, site and token: the
  // 24th arrival runs the token's finish and resets the count for the next step.
  unsigned mhc_done[kStepMaxLayers][2][32];
  unsigned pool_done[kStepMaxLayers][kStepMaxPhases];  // kPoolFp8Matrix with slabs: compute units of the pool completed this step, by layer and pool index
  unsigned reduce_done[kStepMaxLayers][kStepMaxPhases];  // and its reduce units completed, by layer and pool index (the fold trigger of a group)
  unsigned acc_done[kStepMaxLayers];      // the accumulate units completed, by layer (the fold trigger of its group)
  unsigned site_done[kStepMaxLayers][2];  // a hyper-connection site's tokens finished, by layer and site (the router's gate)
  unsigned fold_done[kStepMaxPhases];     // the chain spun the group's fold, by the group's published phase (the gate of the units behind it; a layer has two groups with folds, so not by layer: 2026-10-31)
  unsigned fold_staged[kStepMaxPhases];   // the fold's stage units complete (the fold units' gate), by the group's published phase
  unsigned fold_folded[kStepMaxPhases];   // its fold units complete: the last sets fold_done
  unsigned fold_slot;                     // the step's ring slot: its row of the fold table
  unsigned open_gen;                      // bumped when a gate opens (a fold spun, a site's token finished, a pool's compute units complete): lead units yield to it
  unsigned long long stamp_begin[kStepMaxPhases];  // the last pass: the phase's publish or latency start
  unsigned long long stamp_end[kStepMaxPhases];    // the last pass: its units done, the fold after it included
  unsigned long long idle_first[kStepMaxPhases];   // the last pass: the first worker to find the phase drained
  unsigned long long busy_ns[kStepMaxPhases];      // the last pass: worker time inside the phase's grabs, summed
  unsigned grabs[kStepMaxPhases];                  // the last pass: grabs of the phase's units
  unsigned long long handout_ns[kStepMaxPhases];   // the last pass: thread 0's time from the loop top to the grab, summed
  unsigned long long lead_in_bytes[kStepMaxPhases]; // the last pass: lead bytes streamed while this phase was the published one
  unsigned long long pf_credit[kStepMaxPhases];     // of the phase's staged bytes, those credited as consumed so far
  unsigned long long pf_staged[kStepMaxPhases];     // bytes the lead actually streamed for the phase's lead units (a yielded unit counts what it read)
  unsigned long long wait_ns[kStepMaxPhases];       // the last pass: workers' time spinning on a group's gate and up units before its down units
  unsigned long long ahead_begin[kStepMaxPhases];   // the last pass: the lead's staged bytes not yet consumed when the chain entered the phase
  unsigned cursor_begin[kStepMaxPhases];            // the last pass: the phase the lead's cursor stood in when the chain entered the phase
  unsigned look_blocked[kStepMaxPhases];            // the last pass: workers that found the cursor past the lookahead while this phase was published
  unsigned long long cls_first[kStepMaxPhases][kStepMaxClasses];  // the last pass: when each class of the published group handed out its first unit
  unsigned long long cls_last[kStepMaxPhases][kStepMaxClasses];   // and when its last unit completed
  unsigned long long cls_last_grab[kStepMaxPhases][kStepMaxClasses];  // and when its last unit was handed out (the window's stagger ends here)
  unsigned long long route_select_ns[kStepMaxLayers];  // the last pass: the chain's select over the router's scores
  unsigned long long route_finish_ns[kStepMaxLayers];  // the last pass: the run list, groups and slots from the picks
  unsigned steps_run;                              // steps this launch completed
  unsigned rows;                                   // the running step's rows (the synthetic router's)
  unsigned top_k;
  unsigned seed;
  unsigned route_pool;  // the synthetic router picks from this many experts of a layer (0: all), 32 reproduces the measured overlap
};

// The CSA2 attention consumer's operands (kPoolAttention): the queries after
// the decode tail, q [rows][local_heads * 512] bf16 (a later latency unit
// writes them in the kernel; the test provides them); the rows' requests and
// positions; the window ring (layer l's rows at ring + l * ring_layer_bytes,
// [max_requests][ring_slots] rows of the fp8_block form, the identity table)
// with each row's slot list [rows][window] and counts; the main caches per
// layer ([entries] rows of the fp4_block form, nullptr on a window-only
// layer) with their entries a block, the paged block table and each row's
// selection [rows][index_topk] and counts; the per-source split partials
// (m, l [rows][n_split][local_heads], c [..][512]) and their group partials
// and tickets (kernels/dsa.hpp kAttnSplitGroup); the rotary table per layer
// for the finish's inverse rotation; the output o [rows][local_heads * 512]
// bf16. The sink is the layer's arena range (the pool's offsets[0]).
constexpr int kStepAttnSources = 2;  // 0 the window ring, 1 the main cache
struct StepAttnParams {
  const uint16_t* q = nullptr;
  const int32_t* req_ids = nullptr;
  const int64_t* pos = nullptr;
  const uint8_t* ring = nullptr;
  unsigned long long ring_layer_bytes = 0;
  const int32_t* ring_table = nullptr;
  int ring_slots = 0;
  int max_requests = 0;
  const int32_t* wlist = nullptr;
  const int32_t* wcounts = nullptr;
  int window = 0;
  const uint8_t* const* main = nullptr;  // per layer
  const int* epb = nullptr;              // per layer: the main cache's entries a block
  const int32_t* block_tables = nullptr;
  int blocks_per_request = 0;
  int blocks_total = 0;
  const int32_t* topk = nullptr;
  const int32_t* counts = nullptr;
  int index_topk = 0;
  float scale = 0.f;
  int local_heads = 0;
  const float* const* inv_freq = nullptr;  // per layer: [32]
  float* m[kStepAttnSources] = {nullptr, nullptr};
  float* l[kStepAttnSources] = {nullptr, nullptr};
  float* c[kStepAttnSources] = {nullptr, nullptr};
  float* mg[kStepAttnSources] = {nullptr, nullptr};
  float* lg[kStepAttnSources] = {nullptr, nullptr};
  float* cg[kStepAttnSources] = {nullptr, nullptr};
  int32_t* tickets[kStepAttnSources] = {nullptr, nullptr};
  uint16_t* o = nullptr;
};

// The step's folds on the bus's one-sided write transport (net/bus_write.cuh;
// MOONSHOT-1 phase 4): the host issues each fold's generation ahead
// (CollectiveBus::step_prepare) into a pinned table by step slot and fold
// index (DevPhase::fold_index); the kernel's stage units lay the group's
// partial out in the generation's staging row and the fold units poll, prove
// and fold the peers' rows in rank order, then apply the residual update the
// reduce or accumulate units carried at world 1. World 1: no table, the fold
// is the identity and the updates stay where they were.
constexpr int kStepMaxFolds = 256;
struct StepFoldDev {
  uint32_t gen = 0;
  uint32_t elems = 0;
  net::BusAllReduceCtl* cell = nullptr;
  net::BusWriteBlockRecord* recs = nullptr;
};
struct StepFoldParams {
  net::BusAllReduceGraphView view;
  int world = 1;
  int my_rank = 0;
  const StepFoldDev* table = nullptr;  // pinned [kStepRingSlots][kStepMaxFolds]
  unsigned long long deadline_cycles = 0;
};

struct StepStreamParams {
  // The protocol (step/protocol.hpp): pinned rings of kStepRingSlots, the
  // device token feed [2][kStepMaxRows] by step parity, the host's sequence.
  volatile StepDescriptor* desc = nullptr;
  volatile StepVerdict* verdict = nullptr;
  int32_t* token_feed = nullptr;
  const volatile uint64_t* host_seq = nullptr;
  int max_steps = 1;           // steps this launch runs at most; a skip ends it sooner
  const uint8_t* const* bases;  // device: a layer's span base, per layer
  const DevLayer* layers;
  const DevPhase* phases;
  const DevSegment* segments;
  const DevPool* pools;
  int n_layers = 0;
  unsigned unit_bytes = 0;
  unsigned lead_units = 0;
  StepCtl* ctl = nullptr;
  unsigned* sink = nullptr;  // device [kStepMaxWorkers * 8]: the synthetic consumers' folds
  // The MXFP4 expert consumers' operands when the plan has consumer pools:
  // x [kStepMaxRows][hidden] bf16, act [slots][inter] bf16, out partials
  // [tokens][hidden][slots_per_token] fp32, y [kStepMaxRows][hidden] bf16.
  uint16_t* x = nullptr;  // [kStepMaxRows][x_stride]: the consumers read a row at their own inner width; a hyper-connection site writes its normed rows
  uint16_t* act = nullptr;
  float* out = nullptr;
  uint16_t* y = nullptr;
  uint16_t* proj = nullptr;     // the projections' outputs, bf16, at each pool's out_offset
  float* proj_part = nullptr;   // the projections' slab partials, fp32, at each pool's part_offset
  int x_stride = 0;             // elements a row of x
  int hidden = 0;
  int inter = 0;
  // The router pool's operands: scores and biased scores [kStepMaxRows][n_experts] fp32 the
  // router units write and the chain's select reads; the select's rule as the production
  // router's (kernels/glm_moe.cu): mode 0 sigmoid, 2 sqrt-softplus, the bias on the key.
  float* scores = nullptr;
  float* biased = nullptr;
  int n_experts = 0;
  // Each router unit's top-k candidates a token over its 16 rows (id, score, biased), at
  // [token * n_experts + unit * top_k + i]: the chain's select merges them (two levels, the same
  // rule at each, so the picks are the production select's) in registers.
  int32_t* cand_id = nullptr;
  float* cand_score = nullptr;
  float* cand_biased = nullptr;
  unsigned long long* unit_trace = nullptr;  // [kStepMaxWorkers][8]: a router unit's stage stamps, the last unit a worker ran (null: off)
  int router_mode = 2;
  float routed_scaling = 1.f;
  int norm_topk = 1;
  // The hyper-connection sites' operands (kernels/step/mhc.cuh), by site parity: the attention
  // site reads streams[0] and its update (on the output projection's reduce units) writes
  // streams[1]; the MoE site reads streams[1] and the accumulate's fused update writes streams[0]. The pre coefficients chain the same way, pre[0]
  // one-hot at a step's start. streams [kStepMaxRows][4][hidden] bf16, pre [kStepMaxRows][4],
  // post [kStepMaxRows][4], comb [kStepMaxRows][16], logits [kStepMaxRows][24], all fp32.
  uint16_t* streams[2] = {nullptr, nullptr};
  float* pre[2] = {nullptr, nullptr};
  float* post = nullptr;
  float* comb = nullptr;
  float* logits = nullptr;
  float hc_eps = 1e-6f;
  int sinkhorn_iters = 20;
  float mhc_norm_eps = 1e-6f;  // the coefficient norm's and the sublayer norm's epsilon
  unsigned consumer_smem = 0;  // dynamic shared memory the consumer units need
  int lookahead_layers = 2;
  unsigned long long lead_cap_bytes = 0;  // 0: no lead
  // The lead's idle budget: when the current phase is a latency phase or a
  // drained stream phase, the memory would otherwise idle until the next
  // stream phase, so the cap grows by that time's worth of bytes at this
  // rate (bytes a microsecond; 0: the cap alone). The extra is consumed by
  // the very next stream phases, so the parked set stays the cap's.
  float lead_bytes_per_us = 0.f;
  unsigned route_pool = 0;  // the synthetic router's window of experts a layer (0: all)
  unsigned max_leaders = 0;               // 0: unbounded
  // A merged phase's staging class (its rows into L2 at the fold's trigger) as a set with the
  // lead: at most stage_workers blocks stream staging units at once (one block streams at 10 to
  // 18 GB/s, so 16 would saturate the memory and leave the other workers free for the gated
  // classes behind the fold), and with stage_lead_pause no worker takes a lead unit while a
  // staging class of the group has units left. Off (2026-11-01): the staging takes 50 to 55 µs
  // from the trigger at any worker count (L2-bound beside the site units), so the pause costs
  // the expert phase that much of the lead's staging, and six rate samples with the set on sat
  // at 79.1 to 80.1 against 80.2 to 80.7 without it; the set pays once the staging fits the
  // fold's window (fewer bytes, or an earlier window). 0: unbounded.
  unsigned stage_workers = 0;
  unsigned stage_lead_pause = 0;
  StepAttnParams attn;                         // the attention consumer's operands (kPoolAttention)
  StepFoldParams fold;                         // the folds on the bus (world > 1)
  unsigned long long deadline_ns = 0;          // every in-step spin's deadline (0: none)
  unsigned long long pending_deadline_ns = 0;  // how long the chain waits for the host's next descriptor (0: forever)
};

// The blocks that can be co-resident at kStepThreads a block and this much
// dynamic shared memory: the flags deadlock when the grid exceeds it.
int step_stream_capacity(unsigned smem_bytes = 0);
// Launches the kernel on workers + 1 blocks with the consumers' shared memory.
cudaError_t launch_step_stream(const StepStreamParams& p, int workers, cudaStream_t stream);
// Whether the kernel holds the MXFP4 consumer for this hidden and intermediate width.
bool step_stream_has_mxfp4_consumer(int hidden, int inter);
// Whether the kernel holds the fp8 projection form for a [rows, k] matrix.
bool step_stream_has_fp8_matrix(int k, int rows);
// The shared memory the MXFP4 consumer needs at this intermediate width.
unsigned step_stream_mxfp4_smem(int inter);
// The dynamic shared memory the consumer pools present need: the MXFP4 forms
// when a routed pool at `inter` exists (0: none), the fp8 forms when `fp8`,
// the hyper-connection units' scratch when `mhc`.
unsigned step_stream_consumer_smem(int inter, bool fp8, bool mhc = false, bool attention = false);
// Whether the kernel has a router unit at this inner width.
bool step_stream_has_router(int k);
// The partials' index: token-major, the slots of one output element in one sector.
__host__ __device__ inline size_t step_out_index(int slot, int row, int hidden, int slots_per_token) {
  return (static_cast<size_t>(slot / slots_per_token) * hidden + static_cast<size_t>(row)) * slots_per_token +
         static_cast<size_t>(slot % slots_per_token);
}
// Outputs an accumulation or reduce unit covers: two a thread. Measured
// 2026-10-31 against 2048 (one wave of units): a unit's time grew with its
// chunk (3.6 to 12 µs; one block an SM cannot hide the loads' latency), so
// the projection drains and the accumulate lengthened and the layer lost
// three points.
constexpr int kStepAccChunk = 512;
// Test fixtures' fills: every listed range [offsets[i], +bytes[i]) of `base`
// with hashed bytes in [lo, lo + span) (kFillBytes), e4m3 normals (kFillE4m3)
// or F32 in [0.5, 2) (kFillF32Scales); and the activations x as small bf16.
enum StepFillMode : unsigned { kFillBytes = 0, kFillE4m3 = 1, kFillF32Scales = 2, kFillBf16 = 3 };
void step_stream_fill_ranges(uint8_t* base, const unsigned long long* offsets, const unsigned long long* bytes, int n, unsigned seed,
                             unsigned lo, unsigned span, cudaStream_t stream, StepFillMode mode = kFillBytes);
void step_stream_init_x(uint16_t* x, int n, unsigned seed, cudaStream_t stream);

}  // namespace dgpp::step
