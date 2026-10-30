#pragma once
// DeepSeek-V4.1-Flash's stream plan (MOONSHOT-1.md section 2.4, phase 2):
// the decode walk of models/dsv41/model.cpp (enqueue_layer, the DSpark
// draft block, the head) as phases, every resident grant of
// models/dsv41/loader.cpp as a range, the CSA2 pool's window rows, index
// keys and selected entries, the Engram table rows, the embeddings and the
// head. Built from the config alone through the loader's layout pass
// (ResidentLayerStream::layout_layer), so the plan and the loader cannot
// disagree about a tensor's bytes or place.
#include <cstdint>

#include "models/dsv41/loader.hpp"
#include "step/plan.hpp"

namespace dgpp::step {

// The shape of the step the plan accounts for.
struct Dsv41StepShape {
  int rows = 1;                // the main walk's rows: the verify rows of the pass
  int draft_rows = 0;          // the draft block's rows (dspark_block_size x request groups); 0: no draft pass
  double distinct_main = -1;   // routed experts a main layer reads a step; < 0: the uniform expectation
  double distinct_draft = -1;  // the same for a draft stage
  int64_t scanned_keys = 0;    // index keys an index-source layer scores a row: the request's compressed entries
  int head_reads = 2;          // the lm head's reads a pass: the verify logits and the draft block's base logits
  LoaderHeadSharding head = LoaderHeadSharding::VocabSharded;
  bool bf12_head = false;      // engine.bf16_weights bf12: the head is read as its 12-bit companion, three quarters of the bf16 bytes
};

// The uniform expectation of distinct experts over `rows` rows picking `k`
// of `experts` each: E (1 - (1 - 1/E)^(rows k)).
double dsv41_expected_distinct(int experts, int rows, int k);

// One layer's plan from a laid-out resident record whose pointers are
// `base` + grant offsets (layout_layer, or a loaded layer with its span's
// base). `span_bytes` is the layer's resident span.
LayerPlan dsv41_layer_plan(const Dsv41TextConfig& cfg, const Dsv41LocalGeometry& geo, const Dsv41LayerResident& r,
                           const void* base, uint64_t span_bytes, const Dsv41StepShape& shape);

// The globals' plan: the embedding rows, the final norm and the head, in
// the loader's grant order (build_globals).
LayerPlan dsv41_globals_plan(const Dsv41TextConfig& cfg, const Dsv41LocalGeometry& geo, const Dsv41StepShape& shape);

// The rank's whole step: every main layer, the draft stages when the shape
// has draft rows, then the globals.
StepPlan dsv41_step_plan(const Dsv41TextConfig& cfg, int rank, int world, const Dsv41StepShape& shape);

}  // namespace dgpp::step
