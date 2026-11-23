#pragma once
// Resident loader for Inkling 2 (docs/inkling2_mini_plan.md sections 2 and
// 3): the family behind the shared ResidentLayerStream. The checkpoint is
// one safetensors file. A layer of weights is loaded once and serves every
// visit of that layer.
//
// What a rank keeps of a layer:
//   q_side   BF16 [lh x (2 x 128 + 16 + 1), hidden], the rank's query heads'
//            rows of wq_du, wr_du, gate_attn and attn_sink_proj stacked in
//            that order (one projection of the query input)
//   k, v     BF16 [lkv x 128 (+ lkv x 4), hidden], the rank's key/value
//            heads' rows of wk_dv / wv_dv, followed on a full-attention
//            layer by the dynamic taps' weight_proj rows
//   o_proj   BF16 [hidden, lh x 128], the rank's query heads' columns
//   experts  every expert on the rank's slice of the intermediate width:
//            w13 rows [2 x lo, 2 x hi) with gate and up interleaved as the
//            checkpoint stores them, w2 columns [lo, hi). NVFP4 bytes are
//            the checkpoint's with the per-expert factor amax / denominator
//            stored as its reciprocal (the divisor the shared NVFP4 kernels
//            apply). MXFP8 payload bytes are the checkpoint's and the
//            scale exponents are widened to their exact FP32 values on the
//            same 1 x 32 grid, the form the shared FP8 kernels read
//   small tensors: clip scales, the relative table, convolution taps and mix
//            queries widened to FP32 (exact). Taps are stored by lag, lag 0
//            the current row. Mix queries are scaled by hidden^-0.5.
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"
#include "loaders/resident_stream.hpp"
#include "loaders/safetensors.hpp"
#include "loaders/weight_build.hpp"
#include "models/inkling2/binding.hpp"
#include "models/inkling2/config.hpp"
#include "models/quant_matrix.hpp"

namespace dgpp {

using Inkling2Residency = LoaderResidency;
using Inkling2HeadSharding = LoaderHeadSharding;

// The loop streams' order everywhere: the controllers, the message queries.
enum Inkling2Stream : int { kInk2StreamAttn = 0, kInk2StreamK = 1, kInk2StreamV = 2, kInk2StreamMlp = 3 };

struct Inkling2AttnResident {
  const uint16_t* q_side = nullptr;  // BF16 [q_side_rows, hidden] (packable)
  const uint16_t* k = nullptr;       // BF16 [kv_rows, hidden] (packable)
  const uint16_t* v = nullptr;       // BF16 [kv_rows, hidden] (packable)
  const uint16_t* o_proj = nullptr;  // BF16 [hidden, lh x 128] (packed columns, packable)
  const float* q_scale = nullptr;    // F32 [lh, 128]
  const float* r_scale = nullptr;    // F32 [lh, d_rel]
  const float* rel_proj = nullptr;   // F32 [d_rel, extent]
  const float* k_taps = nullptr;     // F32 [lkv x 128, 4] by lag (the static part on a full layer)
  const float* v_taps = nullptr;
  bool local = true;                 // sliding window, static K/V taps
  int extent = 0;                    // the relative table's width
  int local_heads = 0;
  int head_begin = 0;
  int local_kv_heads = 0;
  int kv_head_begin = 0;
  int64_t q_side_rows = 0;           // lh x (2 x 128 + d_rel + 1)
  int64_t kv_rows = 0;               // lkv x 128, plus lkv x 4 on a full layer
};

struct Inkling2ControllerResident {
  const uint16_t* down = nullptr;         // BF16 [gate_rank, 2 x hidden] (packable)
  const uint16_t* gain_forget = nullptr;  // BF16 [2 x hidden, gate_rank]: gain rows, then forget rows
};

struct Inkling2MoeResident {
  const uint16_t* router = nullptr;    // BF16 [E, hidden]
  const float* router_bias = nullptr;  // F32 [E]
  bool mxfp8 = false;
  // [E x 3] in the shared view-table order: entry 0 is w13 [2 x I/W,
  // hidden], entry 1 is unset, entry 2 is w2 [hidden, I/W]. Exactly one
  // table is filled.
  std::vector<GlmFp4Matrix> fp4;
  std::vector<GlmQuantMatrix> fp8;
  int64_t local_inter = 0;
};

struct Inkling2LayerResident {
  int layer = -1;
  bool local = true;
  bool core = false;
  const uint16_t* attn_norm[3] = {nullptr, nullptr, nullptr};  // BF16 [hidden]: q, k, v (one tensor thrice on a local layer)
  const uint16_t* mlp_norm = nullptr;
  const float* attn_queries = nullptr;  // F32 [attn_query_count, hidden], scaled (null on layer 0)
  int attn_query_count = 0;             // 1 on a local layer, 3 on a full-attention layer
  const float* mlp_query = nullptr;     // F32 [hidden], scaled
  Inkling2AttnResident attn;
  const float* attn_conv = nullptr;     // F32 [hidden, 4] by lag
  const float* mlp_conv = nullptr;
  Inkling2MoeResident moe;
  Inkling2ControllerResident ctrl[4];   // core layers, by Inkling2Stream
  size_t bytes = 0;
};

struct Inkling2GlobalsResident {
  const uint16_t* embed = nullptr;        // BF16 [vocab, hidden], the real rows
  const uint16_t* embed_norm = nullptr;   // BF16 [hidden]
  const uint16_t* final_norm = nullptr;   // BF16 [hidden]
  const uint16_t* unembed = nullptr;      // BF16 [lm_vocab_count, hidden]
  const float* output_query = nullptr;    // F32 [hidden], scaled
  const float* message_queries = nullptr; // F32 [core_layers, 4, hidden], scaled, by Inkling2Stream
  int lm_vocab_begin = 0;
  int lm_vocab_count = 0;
  size_t bytes = 0;
};

// The local TP geometry at (rank, world).
struct Inkling2LocalGeometry {
  int world = 1, rank = 0;
  int local_heads = 0, head_begin = 0;
  int local_kv_heads = 0, kv_head_begin = 0;
  int64_t local_inter = 0;   // intermediate_size / W
  int64_t inter_begin = 0;
  int lm_vocab_begin = 0, lm_vocab_count = 0;
  static Inkling2LocalGeometry from_config(const Inkling2Config& cfg, int rank, int world, Inkling2HeadSharding head);
};

struct Inkling2LoaderFamily {
  using Config = Inkling2Config;
  using Expected = Inkling2ExpectedTensor;
  using LayerResident = Inkling2LayerResident;
  using GlobalsResident = Inkling2GlobalsResident;
  using Geometry = Inkling2LocalGeometry;
  using PresentMap = std::unordered_map<std::string, Inkling2TensorDesc>;
  struct Builder;  // models/inkling2/loader.cpp
  static const char* who() { return "inkling2 loader"; }
  static uint64_t loader_format() { return 1; }
  static int max_layer(const Config& c) { return c.num_hidden_layers; }
  static int main_layers(const Config& c) { return c.num_hidden_layers; }
  static std::vector<Expected> layer_table(const Config& c, int layer) { return inkling2_expected_layer_tensors(c, layer); }
  static std::vector<Expected> global_table(const Config& c) { return inkling2_expected_global_tensors(c); }
  static void validate_binding(const Config& c, const PresentMap& present);
  static void check_sources(const Config&, const LoaderTensorMap&) {}
  static bool digest_included(const Expected& e);
  static bool discard_after_pack(const Expected& e);
  static void build_globals(const Config& c, const Geometry& geo, const LoaderTensorMap& tensors, LayerBump& bump,
                            GlobalsResident& out, uint64_t& source_bytes, uint64_t& verbatim_bytes,
                            LoaderHeadSharding head);
  static size_t globals_bytes(const Config& c, int rank, int world, LoaderHeadSharding head);
  static size_t globals_side_bytes(const Config& c, int rank, int world, LoaderHeadSharding head);
  static size_t extra_resident_bytes(const Config&, int, int) { return 0; }
  static size_t min_staging_bytes() { return 0; }
  static void after_restore(const Config&, int, const LoaderTensorMap&, LayerResident&) {}
};

extern template class ResidentLayerStream<Inkling2LoaderFamily>;

class Inkling2LayerStream : public ResidentLayerStream<Inkling2LoaderFamily> {
 public:
  Inkling2LayerStream(const Inkling2Config& cfg, const std::string& checkpoint_dir, int rank = 0, int world = 1,
                      Inkling2Residency residency = Inkling2Residency::Streaming,
                      Inkling2HeadSharding head = Inkling2HeadSharding::Full);
  ~Inkling2LayerStream() override = default;

  static void set_resident_image_dir(const std::string& dir);
  static const std::string& resident_image_dir();

 protected:
  const std::string& image_dir() const override { return resident_image_dir(); }
};

}  // namespace dgpp
