#pragma once
// The step engine's stream plan (MOONSHOT-1.md section 2.4): every byte one
// rank's decode step reads, as ranges carrying what the streamer, the arena,
// the replay model and the byte meter need to know about them. A family's
// generator (src/step/<family>/plan.cpp) produces a plan from the loader's
// layout of each layer and the family's step structure. Nothing here
// touches the device: a plan is host data, built from a config alone.
//
// A range is one grant of the loader (a payload, a scale grid, a vector) or
// one source outside the resident weights (KV rows, index keys, gathered
// table rows). Resident ranges are byte offsets into the layer's resident
// span in the loader's grant order, the staging mirror's and the image's
// layout, and a layer's resident ranges tile its span exactly (plan_check),
// so a generator cannot miss a tensor silently. Each range says how its
// bytes are decoded (format), ordered (layout), where the streamer lands
// them (staging), when its address is known (dependency), which site reads
// it in which phase, and what a step reads of it: `read_bytes` of its
// `bytes`, `reads` times. Router-bound ranges belong to one expert
// (`group`); a step reads the layer's `expected_groups` of `groups`.
#include <cstdint>
#include <string>
#include <vector>

namespace dgpp::step {

// How a consumer decodes the bytes.
enum class Format : uint8_t {
  Bf16,
  F32,
  Fp8E4m3,      // e4m3 codes, one byte a weight; the scale grid is its own range
  ScaleF32,     // an F32 block-scale grid
  ScaleE8m0,    // an e8m0 scale grid, one byte a block
  Fp4E2m1,      // e2m1 codes, two a byte; the scale grid is its own range
  Exl3Trellis,  // EXL3 trellis tiles; suh and svh are Bf16 ranges
  KvFp4Block,   // KV latent rows: e2m1 codes and e4m3 block scales
  KvFp8Block,   // KV latent rows: e4m3 codes and e8m0 block scales
  IndexKey,     // index keys: e4m3 codes with an F32 scale a key
  Bf12,         // the lossless 12-bit companion of a bf16 matrix
};

// The order of the bytes inside a range.
enum class Layout : uint8_t {
  RowMajor,  // the checkpoint's row-major form
  Tile16,    // 16-row tiles with a tile's 32-code chunks contiguous (section 1.3's measured form)
};

// Where the streamer lands the bytes.
enum class Staging : uint8_t { Register, Shared, L2, Direct };

// When the range's address is known (section 2.4).
enum class Dependency : uint8_t {
  Fixed,      // at load
  Token,      // at step start: embeddings, window rows, index keys, gathered tables
  Router,     // when the layer's router has written its run list
  Selection,  // when the selection the layer attends with has finished
};

// What the bytes are a part of.
enum class Source : uint8_t {
  Resident,  // the layer's resident weights: `offset` is into its span
  Pool,      // the KV and state pool, addressed per request at step time
  Host,      // host memory the step gathers from
};

// The site that reads the range.
enum class Consumer : uint8_t {
  Norm,
  HyperConn,
  Embedding,
  Projection,
  Indexer,
  Compressor,
  Attention,
  Router,
  ExpertGateUp,
  ExpertDown,
  SharedGateUp,
  SharedDown,
  Head,
  DraftHead,
  TableGather,
  None,
};

enum class PhaseKind : uint8_t { Latency, Stream };

// One phase of a layer's walk, in order. `resolves` names the dependency
// class whose addresses are known once the phase has finished (-1: none).
struct Phase {
  std::string name;
  PhaseKind kind = PhaseKind::Stream;
  int resolves = -1;
  // The chain's modeled time: what a latency phase costs on the critical
  // path, and the fold that follows the phase, in microseconds. The
  // streamer's gate interleaves these until the fused chain of section 2.9
  // replaces them with the real kernels.
  float latency_us = 0;
  float fold_after_us = 0;
};

struct Range {
  std::string name;  // the tensor as the family names it, "attn.wq_b.payload"
  Source source = Source::Resident;
  uint64_t offset = 0;      // Resident: the byte offset in the layer's span
  uint64_t bytes = 0;       // the range's bytes; a grant's 256-byte pad is not part of them
  uint64_t read_bytes = 0;  // the bytes one read takes; 0 means all of `bytes`
  uint32_t reads = 1;       // reads a step at the plan's shape; 0: resident, never read by a decode step
  Format format = Format::Bf16;
  Layout layout = Layout::RowMajor;
  Staging staging = Staging::Register;
  Dependency dep = Dependency::Fixed;
  Consumer consumer = Consumer::None;
  int phase = -1;     // the LayerPlan phase that reads it
  int64_t rows = 0;   // the geometry in elements: a matrix's rows and columns, a vector's 1 x n,
  int64_t cols = 0;   // a pool or host range's rows a read and bytes a row
  int scale_of = -1;  // a scale grid: the index of the payload range it scales
  int group = -1;     // Router: the expert index
  int matrix = -1;    // an expert's matrix: 0 gate, 1 up, 2 down

  uint64_t bytes_per_read() const { return read_bytes ? read_bytes : bytes; }
};

struct LayerPlan {
  int layer = -1;              // -1: the globals
  uint64_t span_bytes = 0;     // the resident span, ResidentLayerStream::layer_bytes
  int groups = 0;              // routed experts a Router-bound range may belong to
  double expected_groups = 0;  // of `groups`, how many a step reads at the plan's shape
  std::vector<Phase> phases;
  std::vector<Range> ranges;
};

struct StepPlan {
  std::string family;
  int rank = 0;
  int world = 1;
  int layout_version = 0;         // 0: the loader's grant-order layout, every range row-major
  std::vector<LayerPlan> layers;  // in walk order: the main layers, then the draft stages
  LayerPlan globals;              // embeddings, the final norm, the head
};

// The bytes a step reads, by dependency class. Router-bound bytes are the
// expectation at the plan's shape, so the totals are real-valued.
struct ByteTotals {
  double fixed = 0;
  double token = 0;
  double router = 0;
  double selection = 0;
  double total() const { return fixed + token + router + selection; }
  ByteTotals& operator+=(const ByteTotals& o);
};
ByteTotals layer_step_bytes(const LayerPlan& lp);
ByteTotals step_bytes(const StepPlan& plan);
// The bytes of `lp`'s ranges read by `consumer` a step (the same expectation).
double consumer_step_bytes(const LayerPlan& lp, Consumer consumer);
double consumer_step_bytes(const StepPlan& plan, Consumer consumer);

// The structural checks: the resident ranges tile [0, span_bytes) in
// 256-byte-aligned grants with no overlap, every range has bytes, a phase
// and a consumer, read_bytes fits, scale grids point at payloads of their
// own group, Router-bound ranges name a group under `groups`. Returns the
// problems found, empty when the plan is sound.
std::vector<std::string> plan_check(const LayerPlan& lp);
std::vector<std::string> plan_check(const StepPlan& plan);

const char* to_string(Format f);
const char* to_string(Layout l);
const char* to_string(Staging s);
const char* to_string(Dependency d);
const char* to_string(Source s);
const char* to_string(Consumer c);

// A readable account of a plan: the step's bytes by dependency class and
// by consumer, and each layer's ranges and span.
std::string plan_summary(const StepPlan& plan);

}  // namespace dgpp::step
