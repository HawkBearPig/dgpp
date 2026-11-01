#pragma once
// DeepSeek-V4.1's arena (MOONSHOT-1.md section 2.4): the resident layout the
// step engine's streamer reads, as a placement for the loader's bump
// allocator (ResidentLayout). Every grant keeps its bytes; the arena is the
// order the step consumes them in: each phase's fixed ranges together, then
// the routed experts of the phase as one pool in expert order, gate and up
// in one pool and down in the next, so a run of consecutive experts is one
// contiguous range for the streamer.
#include <cstdint>
#include <vector>

#include "loaders/resident_stream.hpp"
#include "models/dsv41/config.hpp"
#include "step/dsv41/plan.hpp"

namespace dgpp::step {

// The layout's version, mixed into the resident image key.
constexpr uint64_t kDsv41ArenaVersion = 1;

// The device offset of every main grant of `layer` in the loader's grant
// order, under the arena. A permutation of the grant-order layout.
std::vector<size_t> dsv41_arena_placement(const Dsv41TextConfig& cfg, int layer, int rank, int world);

// The arena as the loader takes it (ResidentLayerStream::set_layout).
ResidentLayout dsv41_resident_layout(const Dsv41TextConfig& cfg, int rank, int world);

// The order the arena places a layer's ranges in: the plan's ranges
// sorted by phase, then class (fixed before routed), then expert, matrix
// and payload before scales, with ties in grant order. Exposed for tests.
std::vector<size_t> dsv41_arena_order(const LayerPlan& lp);

}  // namespace dgpp::step
