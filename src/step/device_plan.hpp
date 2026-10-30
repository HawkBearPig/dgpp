#pragma once
// The stream plan lowered for the streamer (MOONSHOT-1.md sections 2.1 to
// 2.6): a layer's phases in walk order, each with its fixed-class bytes as
// a few contiguous segments of the arena, its router-bound bytes as one
// pool of equal-stride groups, and the chain's modeled latencies. Units are
// fixed-size byte runs the workers grab by atomic counter; the fixed units
// of every phase form one global sequence the lead walks ahead of the
// chain. Host data, uploaded as flat tables.
#include <cstdint>
#include <string>
#include <vector>

#include "step/plan.hpp"

namespace dgpp::step {

struct DevSegment {
  uint64_t offset = 0;  // into the layer's device span
  uint64_t bytes = 0;
  uint32_t unit_begin = 0;  // the phase's first unit of this segment
  uint32_t units = 0;
};

// A pool's consumer: how its units are cut and computed.
enum DevPoolKind : uint8_t {
  kPoolBytes = 0,        // synthetic: units of unit_bytes, streamed and folded
  kPoolMxfp4GateUp = 1,  // MXFP4 gate and up pair: units of 128 output rows a group
  kPoolMxfp4Down = 2,    // MXFP4 down: units of 128 output rows a group
  kPoolFp8GateUp = 3,    // the shared expert's fp8 pair, row-major with F32 32 x 32 scales: one group of every token
  kPoolFp8Down = 4,      // the shared expert's fp8 down
  kPoolMxfp4RowGateUp = 5,  // MXFP4 gate and up over the checkpoint's row-major layout, through the L2 bridge
  kPoolMxfp4RowDown = 6,    // MXFP4 down, row-major
};
#if defined(__CUDACC__)
#define DGPP_STEP_PLAN_HD __host__ __device__
#else
#define DGPP_STEP_PLAN_HD
#endif
DGPP_STEP_PLAN_HD inline bool pool_is_gate_up(uint8_t kind) {
  return kind == kPoolMxfp4GateUp || kind == kPoolFp8GateUp || kind == kPoolMxfp4RowGateUp;
}
DGPP_STEP_PLAN_HD inline bool pool_is_down(uint8_t kind) { return kind == kPoolMxfp4Down || kind == kPoolFp8Down || kind == kPoolMxfp4RowDown; }

struct DevPool {
  uint64_t base = 0;    // group 0's first byte
  uint64_t stride = 0;  // group g begins at base + g * stride
  uint64_t bytes = 0;   // the bytes of one group
  int32_t groups = 0;
  int32_t units_per_group = 0;
  uint8_t kind = kPoolBytes;
  uint8_t pad[3] = {0, 0, 0};
  int32_t rows = 0;     // a consumer pool: the matrix's output rows (gate and up: I; down: H)
  int32_t k = 0;        // and its inner width (gate and up: H; down: I)
  int32_t grab = 1;     // units a worker takes at once: consecutive units of one group share their staged activations
  // A consumer pool's matrices inside a group: gate payload, gate scales,
  // up payload, up scales (gate_up), or down payload, down scales (down).
  uint64_t offsets[4] = {0, 0, 0, 0};
};

struct DevPhase {
  uint8_t kind = 0;        // 0 latency, 1 stream
  uint8_t resolves = 0xff; // a Dependency value the phase's end resolves (0xff none)
  uint8_t waits_host = 0;  // the phase reads rows a host gather publishes (section 2.11): it waits on the host's sequence
  uint8_t accumulate = 0;  // the ordered accumulation of the routed partials as worker units
  // The next phase runs inside this one: its units follow this phase's in
  // the grab order, and a down unit of group g waits for the group's gate
  // and up units, so the pair's drain tail is filled with down work.
  uint8_t merged_next = 0;
  float latency_us = 0;    // a latency phase's modeled time
  float fold_after_us = 0; // the fold that follows the phase
  int32_t seg_begin = 0;   // fixed segments
  int32_t seg_count = 0;
  uint32_t fixed_units = 0;
  // Lead-only segments: fixed-class bytes a consumer pool reads (the shared
  // expert's), which the lead prefetches into L2 but no worker streams.
  int32_t lead_seg_begin = 0;
  int32_t lead_seg_count = 0;
  uint32_t lead_units = 0;  // fixed_units plus the lead-only units: the phase's share of the lead sequence
  int32_t pool = -1;       // index into pools, or -1: the routed experts
  int32_t pool2 = -1;      // a second pool: the shared expert, fixed-class, one group of every token
  uint32_t lead_begin = 0; // this phase's first unit in the global lead sequence
  int32_t layer = 0;       // the layer (index into DevicePlan::layers) the phase belongs to
};

struct DevLayer {
  int32_t phase_begin = 0;
  int32_t phase_count = 0;
  uint64_t span_bytes = 0;
};

struct DevicePlan {
  uint32_t unit_bytes = 0;
  std::vector<DevLayer> layers;
  std::vector<DevPhase> phases;
  std::vector<DevSegment> segments;
  std::vector<DevPool> pools;
  uint32_t lead_units = 0;  // fixed units over the whole plan
  // Bytes a step reads at the plan's shape, by class, for the rate.
  double fixed_bytes = 0;
  double router_bytes = 0;
};

// Lowers the layers of a plan whose offsets are the arena's (dsv41_layer_plan
// over a placed layout). Consecutive resident ranges of one phase and class
// coalesce into a segment; a phase's router-bound ranges must form equal
// groups at one stride, or the lowering throws. Ranges outside the resident
// span (KV rows, gathered tables) are not streamed from the arena and are
// left out. `unit_bytes` is the grab size, a multiple of 512. With
// `consumers`, the MXFP4 expert pools take the consumer kinds and the
// accumulate phase becomes worker units; without, every pool is bytes.
DevicePlan lower_plan(const std::vector<LayerPlan>& layers, uint32_t unit_bytes, bool consumers = false);

std::string device_plan_summary(const DevicePlan& dp);

}  // namespace dgpp::step
