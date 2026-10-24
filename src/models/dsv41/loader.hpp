#pragma once
// DeepSeek-V4.1-Flash weight loader using ResidentLayerStream (2026-09-13,
// docs/deepseek_v41_flash_plan.md G3). The family supplies tensor
// bindings, TP geometry and the builders for the release's own formats —
// fp8 e4m3 with e8m0 scales on 32 x 32 blocks (converted to the fp32 grid
// the fp8 core reads, plan D2), MXFP4 experts (e2m1 + e8m0 per 32, kept
// as they are), the mHC coefficient matrices rounded fp32 -> bf16 at load
// (plan D10), the Engram tables mmap'ed from their shards (plan D5). The
// shared stream owns allocations, staging, resident images, byte
// accounting and digests.
//
// Placement (every slice a formula in the world size W, plan §2):
//   attention: wq_a and wkv replicated; wq_b rows per head block (64/W
//              heads); wo_a rows per output group (8/W groups); wo_b
//              packed columns of those groups; attn_sink per local head;
//              the norms replicated.
//   indexer:   replicated (wq_b, weights_proj; wk + k_norm on kv sources).
//   compressor: replicated (kv sources).
//   MoE:       router and its bias replicated; every routed expert and the
//              shared expert sliced on the intermediate dim at 2304/W
//              (w1/w3 rows, w2 columns on a 32-block boundary).
//   mHC:       replicated (fn as bf16, base / scale fp32).
//   Engram:    wkv's K columns of this rank's hash heads (8/W per n-gram
//              size, three ranges), q_weight / k_weight replicated; the
//              tables stay in their shards, row-sharded by hash head.
//   draft:     as a main layer; main_proj, main_norm, norm, the Markov
//              embedding, the Markov head and the confidence head
//              replicated (a layer's bytes cannot depend on the head
//              sharding — the stream's counting pass has none — so the
//              [vocab, rank] head is held whole, 66 MB per rank, and its
//              lm-head rows are addressed at use).
//   globals:   embed replicated (a row gather) or vocab-sharded, the
//              final norm, head vocab-sharded under VocabSharded.
#include <cstdint>
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
#include "models/dsv41/binding.hpp"
#include "models/dsv41/config.hpp"
#include "models/exl3_matrix.hpp"
#include "models/quant_matrix.hpp"

namespace dgpp {

using Dsv41Residency = LoaderResidency;
using Dsv41HeadSharding = LoaderHeadSharding;
using Dsv41ReplicatedDigest = ReplicatedDigest;

constexpr int kDsv41Fp8Block = 32;  // the release's fp8 grid (both axes)

struct Dsv41AttnResident {
  GlmQuantMatrix wq_a;   // fp8 [q_lora, hidden], replicated
  GlmQuantMatrix wkv;    // fp8 [head_dim, hidden], replicated
  GlmQuantMatrix wq_b;   // fp8 [local_heads * head_dim, q_lora]
  GlmQuantMatrix wo_a;   // fp8 [local_groups * o_lora, heads_per_group * head_dim]
  GlmQuantMatrix wo_b;   // fp8 [hidden, local_groups * o_lora] (packed columns)
  const uint16_t* q_norm = nullptr;   // bf16 [q_lora]
  const uint16_t* kv_norm = nullptr;  // bf16 [head_dim]
  const float* attn_sink = nullptr;   // f32 [local_heads]
  // The indexer (index sources) and the index keys (kv sources).
  GlmQuantMatrix idx_wq_b;              // fp8 [index_heads * 128, q_lora]
  const uint16_t* idx_wp = nullptr;     // bf16 [index_heads, hidden]
  const uint16_t* idx_wk = nullptr;     // bf16 [128, head_dim] (kv sources)
  const uint16_t* idx_k_norm = nullptr; // bf16 [128] (kv sources)
  // The compressor (kv sources; wgate at ratio > 1).
  const uint16_t* comp_wkv = nullptr;   // bf16 [head_dim, hidden]
  const uint16_t* comp_wgate = nullptr; // bf16 [head_dim, hidden]
  const uint16_t* comp_norm = nullptr;  // bf16 [head_dim]
  // The EXL3 pack's projections (docs/deepseek_v41_flash_exl3_plan.md §3.1;
  // the fp8 / bf16 members above stay unset): the same slices as the fp8
  // ones — wq_b's rows per head block, wo_a one matrix per LOCAL group
  // (the group's heads x head_dim -> o_lora), wo_b's packed input columns
  // of those groups; the compressors and the indexer's wq_b / wk EXL3 too.
  Exl3Matrix wq_a_x, wkv_x, wq_b_x, wo_b_x;
  std::vector<Exl3Matrix> wo_a_x;  // [local_groups]
  Exl3Matrix idx_wq_b_x, idx_wk_x, comp_wkv_x, comp_wgate_x;
  int local_heads = 0;
  int head_begin = 0;
  int local_groups = 0;
  int group_begin = 0;
  bool exl3() const { return wq_a_x.tiles != nullptr; }
  bool index_source() const { return idx_wq_b.payload != nullptr || idx_wq_b_x.tiles != nullptr; }
  bool kv_source() const { return comp_wkv != nullptr || comp_wkv_x.tiles != nullptr; }
};

struct Dsv41MoeResident {
  const uint16_t* router = nullptr;     // bf16 [E, hidden]
  const float* router_bias = nullptr;   // f32 [E]
  GlmQuantMatrix shared[3];             // fp8: w1, w3 [S/W, hidden]; w2 [hidden, S/W]
  std::vector<GlmFp4Matrix> experts;    // [E * 3]: w1, w3, w2 per expert (MXFP4, inter-sliced)
  // The EXL3 pack's experts: [E * 3] gate, up, down per expert and the
  // shared triple, sliced on the intermediate in whole 128-blocks (the
  // uneven split alternating by parity, exl3_expert_block_split: a
  // matrix's own rows / cols carry its width, local_inter the widest);
  // the fp4 / fp8 members stay unset.
  std::vector<Exl3Matrix> experts_x;
  Exl3Matrix shared_x[3];
  bool exl3() const { return !experts_x.empty(); }
  int n_experts = 0;
  int64_t local_inter = 0;              // this rank's intermediate slice (I/W, or the EXL3 block split's widest)
  int64_t local_shared_inter = 0;       // this rank's shared slice
  // The view-table order the MoE layer expects: gate (w1), up (w3), down (w2).
  const GlmFp4Matrix& expert(int e, int i) const { return experts[static_cast<size_t>(e) * 3 + i]; }
};

struct Dsv41MhcResident {
  const uint16_t* attn_fn = nullptr;  // bf16 [24, 4 * hidden] (fp32 in the file, rounded at load)
  const float* attn_base = nullptr;   // f32 [24]
  const float* attn_scale = nullptr;  // f32 [3]
  const uint16_t* ffn_fn = nullptr;
  const float* ffn_base = nullptr;
  const float* ffn_scale = nullptr;
};

struct Dsv41EngramResident {
  bool present = false;
  int table_index = -1;                 // ordinal among the Engram layers
  GlmQuantMatrix wkv;                   // fp8 [(hc + 1) * hidden, rows_local * head_dim]: this rank's K columns
  Exl3Matrix wkv_x;                     // the EXL3 pack's (the same K columns)
  const uint16_t* q_weight = nullptr;   // bf16 [hc, hidden]
  const uint16_t* k_weight = nullptr;   // bf16 [hc, hidden]
};

struct Dsv41DraftResident {
  GlmQuantMatrix main_proj;                 // stage 0: fp8 [hidden, targets * hidden]
  Exl3Matrix main_proj_x;                   // stage 0, the EXL3 pack
  const uint16_t* main_norm = nullptr;      // stage 0: bf16 [hidden]
  const uint16_t* norm = nullptr;           // last stage: bf16 [hidden]
  const uint16_t* markov_embed = nullptr;   // last stage: bf16 [vocab, rank], replicated
  const uint16_t* markov_head = nullptr;    // last stage: bf16 [vocab, rank], replicated
  const float* confidence = nullptr;        // last stage: f32 [hidden + rank]
};

struct Dsv41LayerResident {
  int layer = -1;
  const uint16_t* attn_norm = nullptr;  // bf16 [hidden]
  const uint16_t* ffn_norm = nullptr;   // bf16 [hidden]
  Dsv41AttnResident attn;
  Dsv41MoeResident moe;
  Dsv41MhcResident mhc;
  Dsv41EngramResident engram;
  Dsv41DraftResident draft;
  size_t bytes = 0;
};

struct Dsv41GlobalsResident {
  const uint16_t* embed = nullptr;       // bf16 [embed_vocab_count, hidden]
  int embed_vocab_begin = 0;
  int embed_vocab_count = 0;
  const uint16_t* final_norm = nullptr;  // bf16 [hidden]
  const uint16_t* lm_head = nullptr;     // bf16 [lm_vocab_count, hidden]
  Exl3Matrix lm_head_x;                  // the EXL3 pack's head rows [lm_vocab_begin, +lm_vocab_count)
  int lm_vocab_begin = 0;
  int lm_vocab_count = 0;
  size_t bytes = 0;
};

// One Engram table kept in its shard (plan D5): the payload tensor
// [rows, head_dim] e4m3 and the scale tensor [rows, head_dim / 32] e8m0,
// mapped read-only with random-access advice, never copied to the device.
// The host gathers the rows a step needs into pinned staging as [n,
// rows_local, head_dim + head_dim / 32] (a row's payload then its scales),
// the staged gather kernel converts them.
class Dsv41EngramTableMmap {
 public:
  Dsv41EngramTableMmap(const std::string& path, uint64_t payload_begin, uint64_t scale_begin,
                       int64_t rows, int head_dim);
  ~Dsv41EngramTableMmap();
  Dsv41EngramTableMmap(const Dsv41EngramTableMmap&) = delete;
  Dsv41EngramTableMmap& operator=(const Dsv41EngramTableMmap&) = delete;
  const uint8_t* payload_row(int64_t row) const;
  const uint8_t* scale_row(int64_t row) const;
  // dst[(t * rows_local + j) * row_bytes ..] = the payload then the scales
  // of row ids[t * ids_stride + sel[j]], for t < n, j < rows_local. Faults
  // the pages in parallel (a thread per 256 rows past the first 256).
  void gather(const int32_t* ids, int64_t ids_stride, const int32_t* sel, int rows_local, int n,
              uint8_t* dst) const;
  int64_t rows() const { return rows_; }
  int head_dim() const { return head_dim_; }
  int row_bytes() const { return head_dim_ + head_dim_ / 32; }
  size_t mapped_bytes() const { return len_; }

 private:
  int fd_ = -1;
  uint8_t* base_ = nullptr;
  size_t len_ = 0;
  uint64_t payload_begin_ = 0;
  uint64_t scale_begin_ = 0;
  int64_t rows_ = 0;
  int head_dim_ = 0;
};

// The Engram tables of a rank: one mapping per Engram layer, and the
// selection of this rank's rows out of a token's id row (ids [rows,
// layers, (max_ngram - 1) * heads]: entry (n, h) at n * heads + h — the
// rank's rows are its heads_local heads of every n-gram size).
struct Dsv41EngramTables {
  std::vector<std::unique_ptr<Dsv41EngramTableMmap>> tables;  // per Engram layer
  int head_dim = 0;
  int heads = 0;
  int heads_local = 0;
  int head_begin = 0;
  int ngrams = 0;
  std::vector<int32_t> sel;  // [ngrams * heads_local] column indices into a token's id row
  int rows_local() const { return ngrams * heads_local; }
  size_t mapped_bytes() const {
    size_t b = 0;
    for (const auto& t : tables) b += t ? t->mapped_bytes() : 0;
    return b;
  }
};

// The local TP geometry at (rank, world): every slice bound the builders
// and the views use, in one place.
struct Dsv41LocalGeometry {
  int world = 1, rank = 0;
  int local_heads = 0, head_begin = 0;      // attention heads
  int local_groups = 0, group_begin = 0;    // wo_a / wo_b output groups
  // The intermediate slices: [inter_begin, +local_inter) of
  // moe_intermediate_size and [shared_inter_begin, +local_shared_inter) of
  // shared_expert_inter() — I/W each under the release format; under EXL3
  // whole 128-blocks: inter_begin is the even-parity slice's begin and
  // local_inter the widest slice (640 at world 4), each expert's own slice
  // being exl3_expert_block_split's (5/5/4/4 blocks on even experts,
  // 4/4/5/5 on odd; the shared expert by the layer's parity).
  int64_t local_inter = 0;
  int64_t inter_begin = 0;
  int64_t local_shared_inter = 0;
  int64_t shared_inter_begin = 0;
  int engram_heads_local = 0, engram_head_begin = 0;
  int lm_vocab_begin = 0, lm_vocab_count = 0;
  int embed_vocab_begin = 0, embed_vocab_count = 0;
  static Dsv41LocalGeometry from_config(const Dsv41TextConfig& cfg, int rank, int world,
                                       Dsv41HeadSharding head);
};

// The family behind the shared stream (loaders/resident_stream.hpp).
struct Dsv41LoaderFamily {
  using Config = Dsv41TextConfig;
  using Expected = Dsv41ExpectedTensor;
  using LayerResident = Dsv41LayerResident;
  using GlobalsResident = Dsv41GlobalsResident;
  using Geometry = Dsv41LocalGeometry;
  using PresentMap = std::unordered_map<std::string, Dsv41TensorDesc>;
  struct Builder;  // models/dsv41/loader.cpp
  static const char* who() { return "dsv41 loader"; }
  // 2 since 2026-10-23: the experts' intermediate slices alternate by parity
  // (exl3_expert_block_split), so an image of the one-rule layout is stale.
  static uint64_t loader_format() { return 2; }
  static int max_layer(const Config& c) { return c.max_layer(); }
  static int main_layers(const Config& c) { return c.num_hidden_layers; }
  static std::vector<Expected> layer_table(const Config& c, int layer) {
    return dsv41_expected_layer_tensors(c, layer);
  }
  static std::vector<Expected> global_table(const Config& c) { return dsv41_expected_global_tensors(c); }
  static void validate_binding(const Config& c, const PresentMap& present);
  static void check_sources(const Config&, const LoaderTensorMap&) {}
  static bool digest_included(const Expected& e);
  static bool discard_after_pack(const Expected&) { return false; }
  static void build_globals(const Config& c, const Geometry& geo, const LoaderTensorMap& tensors,
                            LayerBump& bump, GlobalsResident& out, uint64_t& source_bytes,
                            uint64_t& verbatim_bytes, LoaderHeadSharding head);
  static size_t globals_bytes(const Config& c, int rank, int world, LoaderHeadSharding head);
  // The Engram tables from another snapshot (the EXL3 pack ships none; the
  // base checkpoint's shards 47 / 48 hold them): the stream admits the
  // table tensors from this directory when the checkpoint's own shards do
  // not carry them (loaders/resident_stream.hpp's extra shard directory).
  // Empty: the checkpoint's own. Set before the stream opens.
  static std::string extra_shard_dir();
  static bool admit_extra_tensor(const std::string& name);
  static size_t extra_resident_bytes(const Config&, int, int) { return 0; }
  static size_t min_staging_bytes() { return 0; }
  static void after_restore(const Config&, int, const LoaderTensorMap&, LayerResident&) {}
};

extern template class ResidentLayerStream<Dsv41LoaderFamily>;

class Dsv41LayerStream : public ResidentLayerStream<Dsv41LoaderFamily> {
 public:
  Dsv41LayerStream(const Dsv41TextConfig& cfg, const std::string& checkpoint_dir, int rank = 0,
                   int world = 1, Dsv41Residency residency = Dsv41Residency::Streaming,
                   Dsv41HeadSharding head = Dsv41HeadSharding::Full, bool resident_mtp = false);
  ~Dsv41LayerStream() override = default;

  static void set_resident_image_dir(const std::string& dir);
  static const std::string& resident_image_dir();
  // The Engram source snapshot (engine.engram_source): where the tables'
  // shards — and the Engram sidecar, when the checkpoint has none — are
  // read from. Empty: the checkpoint directory.
  static void set_engram_source_dir(const std::string& dir);
  static const std::string& engram_source_dir();
  // The deployment's `engine.embed_sharding` ("vocab": each rank holds its
  // lm-head slice of the embedding; "replicated": the whole table).
  static void set_embed_vocab_sharded(bool on);
  static bool embed_vocab_sharded();

  // The Engram tables mmap'ed from their shards (plan D5): built on the
  // first call from the shard mappings, so it must precede
  // release_sources(); the mappings outlive the release.
  const Dsv41EngramTables& load_engram_tables();

 protected:
  const std::string& image_dir() const override { return resident_image_dir(); }

 private:
  Dsv41EngramTables engram_{};
  bool engram_loaded_ = false;
};

}  // namespace dgpp
