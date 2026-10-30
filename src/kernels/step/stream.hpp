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

#include "step/device_plan.hpp"
#include "step/protocol.hpp"

namespace dgpp::step {

constexpr int kStepMaxPhases = 1024;
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
  unsigned pad2[30];
  unsigned long long pf_bytes;           // bytes the lead has read
  unsigned long long consumed_pf_bytes;  // of them, the bytes of phases now complete
  unsigned pad3[28];
  unsigned next[kStepMaxPhases];
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
  unsigned pool_done[kStepMaxPhases];  // kPoolFp8Matrix with slabs: compute units of the pool completed this step, by pool index
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
  unsigned steps_run;                              // steps this launch completed
  unsigned rows;                                   // the running step's rows (the synthetic router's)
  unsigned top_k;
  unsigned seed;
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
  const uint16_t* x = nullptr;  // [kStepMaxRows][x_stride]: the consumers read a row at their own inner width
  uint16_t* act = nullptr;
  float* out = nullptr;
  uint16_t* y = nullptr;
  uint16_t* proj = nullptr;     // the projections' outputs, bf16, at each pool's out_offset
  float* proj_part = nullptr;   // the projections' slab partials, fp32, at each pool's part_offset
  int x_stride = 0;             // elements a row of x
  int hidden = 0;
  int inter = 0;
  unsigned consumer_smem = 0;  // dynamic shared memory the consumer units need
  int lookahead_layers = 2;
  unsigned long long lead_cap_bytes = 0;  // 0: no lead
  // The lead's idle budget: when the current phase is a latency phase or a
  // drained stream phase, the memory would otherwise idle until the next
  // stream phase, so the cap grows by that time's worth of bytes at this
  // rate (bytes a microsecond; 0: the cap alone). The extra is consumed by
  // the very next stream phases, so the parked set stays the cap's.
  float lead_bytes_per_us = 0.f;
  unsigned max_leaders = 0;               // 0: unbounded
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
// The partials' index: token-major, the slots of one output element in one sector.
__host__ __device__ inline size_t step_out_index(int slot, int row, int hidden, int slots_per_token) {
  return (static_cast<size_t>(slot / slots_per_token) * hidden + static_cast<size_t>(row)) * slots_per_token +
         static_cast<size_t>(slot % slots_per_token);
}
constexpr int kStepAccChunk = 512;  // outputs an accumulation unit folds
// Test fixtures' fills: every listed range [offsets[i], +bytes[i]) of `base`
// with hashed bytes in [lo, lo + span) (kFillBytes), e4m3 normals (kFillE4m3)
// or F32 in [0.5, 2) (kFillF32Scales); and the activations x as small bf16.
enum StepFillMode : unsigned { kFillBytes = 0, kFillE4m3 = 1, kFillF32Scales = 2 };
void step_stream_fill_ranges(uint8_t* base, const unsigned long long* offsets, const unsigned long long* bytes, int n, unsigned seed,
                             unsigned lo, unsigned span, cudaStream_t stream, StepFillMode mode = kFillBytes);
void step_stream_init_x(uint16_t* x, int n, unsigned seed, cudaStream_t stream);

}  // namespace dgpp::step
