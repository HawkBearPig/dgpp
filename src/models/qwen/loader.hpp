#pragma once
// Qwen3.8-Flash-Next weight loader. The family defines tensor bindings,
// TP geometry, builders and globals; ResidentLayerStream owns allocations,
// staging, resident images, byte accounting and digests. Layers are built
// directly at the rank's geometry. Dense projections use checkpoint BF16
// or load-time FP8; routed experts use the checkpoint's FP8 or NVFP4 format.
// The n-gram table can be resident or gathered from a read-only mapping.
//
// Placement (every slice a formula in the world size W):
//   GDN: 16/W key heads and 48/W value heads per rank — in_proj_qkv rows
//        per segment, the conv channels alike, in_proj_z rows, in_proj_a/b
//        rows, A_log/dt_bias (widened to F32), out_proj packed columns;
//        the head norm replicated.
//   QSA: 24/W query heads (q_proj keeps each head's [q | gate] rows
//        together), one kv head per rank (its k/v rows; a head shared by
//        W/2 ranks at W > 2), o_proj packed columns; the q/k norms and the
//        whole indexer replicated.
//   MoE: router and shared gate replicated; the shared expert BF16 gate/up
//        rows and packed down columns at S/W; every routed expert's e4m3
//        gate/up rows and down columns at I/W on a gcd(128, I/W)-row scale
//        grid (the parent 128-block's scale replicated: exact).
//   GR:  every site replicated (measured, D3).
//   PLE: key/value projections K-sliced to this rank's 16/W hash heads'
//        columns (packed); norms, conv, buffers and scale replicated. The
//        n-gram table is not a layer: load_ngram_table() puts this rank's
//        contiguous row range (its heads) into its own device allocation,
//        straight from the shards through the staging mirror.
//   Globals: embed replicated (a row gather), lm_head vocab-sharded, the
//        final mixer and the draft head's own tensors replicated.
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
#include "models/qwen/binding.hpp"
#include "models/qwen/config.hpp"
#include "models/quant_matrix.hpp"

namespace dgpp {

using QwenResidency = LoaderResidency;
using QwenHeadSharding = LoaderHeadSharding;
using QwenReplicatedDigest = ReplicatedDigest;

struct QwenGrResident {
  const uint16_t* hc_norm = nullptr;  // BF16 [W]
  const uint16_t* down = nullptr;     // BF16 [r, W]
  const uint16_t* up = nullptr;       // BF16 [W, r]
  const uint16_t* inject = nullptr;   // BF16 [n, W]
  // engine.dense_weights = "fp8": the same two in block FP8,
  // the BF16 pointers null (loaders/fp8_quant.hpp).
  GlmQuantMatrix down_fp8, up_fp8;
};

struct QwenGdnResident {
  const uint16_t* in_proj_qkv = nullptr;  // BF16 [lk*dk | lk*dk | lv*dv, hidden]
  const uint16_t* conv = nullptr;         // BF16 [lk*dk + lk*dk + lv*dv, width]
  const uint16_t* in_proj_z = nullptr;    // BF16 [lv*dv, hidden]
  const uint16_t* in_proj_a = nullptr;    // BF16 [lv, hidden]
  const uint16_t* in_proj_b = nullptr;    // BF16 [lv, hidden]
  GlmQuantMatrix in_proj_qkv_fp8, in_proj_z_fp8, out_proj_fp8;  // dense_weights fp8 (a, b stay BF16)
  const float* a_log = nullptr;           // F32 [lv]
  const float* dt_bias = nullptr;         // F32 [lv]
  const uint16_t* norm = nullptr;         // BF16 [dv]
  const uint16_t* out_proj = nullptr;     // BF16 [hidden, lv*dv] (packed columns)
  int local_key_heads = 0;
  int local_value_heads = 0;
};

struct QwenQsaResident {
  const uint16_t* q_proj = nullptr;         // BF16 [lh * 2 * d, hidden]
  const uint16_t* k_proj = nullptr;         // BF16 [lkv * d, hidden]
  const uint16_t* v_proj = nullptr;         // BF16 [lkv * d, hidden]
  const uint16_t* o_proj = nullptr;         // BF16 [hidden, lh * d] (packed columns)
  const uint16_t* q_norm = nullptr;         // BF16 [d]
  const uint16_t* k_norm = nullptr;         // BF16 [d]
  const uint16_t* index_qk_proj = nullptr;  // BF16 [(nH + 1) * di, hidden]
  GlmQuantMatrix q_proj_fp8, k_proj_fp8, v_proj_fp8, o_proj_fp8, index_qk_proj_fp8;  // dense_weights fp8
  const uint16_t* index_q_norm = nullptr;   // BF16 [di]
  const uint16_t* index_k_norm = nullptr;   // BF16 [di]
  int local_heads = 0;      // query heads on this rank
  int head_begin = 0;       // first global query head
  int local_kv_heads = 0;   // kv heads on this rank (1 at W >= kv heads)
  int kv_head_begin = 0;    // first global kv head
};

// Qwen3.5 full-attention layers: plain GQA (no indexer). Same q/k/v/o +
// q/k norms as QSA, the [q | gate] interleave, dense causal attention.
struct QwenFullAttnResident {
  const uint16_t* q_proj = nullptr;  // BF16 [lh * 2 * d, hidden]
  const uint16_t* k_proj = nullptr;  // BF16 [lkv * d, hidden]
  const uint16_t* v_proj = nullptr;  // BF16 [lkv * d, hidden]
  const uint16_t* o_proj = nullptr;  // BF16 [hidden, lh * d] (packed columns)
  const uint16_t* q_norm = nullptr;  // BF16 [d]
  const uint16_t* k_norm = nullptr;  // BF16 [d]
  GlmQuantMatrix q_proj_fp8, k_proj_fp8, v_proj_fp8, o_proj_fp8;  // dense_weights fp8
  int local_heads = 0;      // query heads on this rank
  int head_begin = 0;       // first global query head
  int local_kv_heads = 0;   // kv heads on this rank
  int kv_head_begin = 0;    // first global kv head
};

struct QwenMoeResident {
  const uint16_t* router = nullptr;       // BF16 [E, hidden]
  const uint16_t* shared_gate = nullptr;  // BF16 [1, hidden]
  const uint16_t* shared[3] = {};         // BF16 gate/up [S/W, hidden], down [hidden, S/W]
  GlmQuantMatrix shared_fp8[3];           // dense_weights fp8: the same three
  std::vector<GlmQuantMatrix> experts;    // gate, up, down per expert (inter-sliced), the FP8 form
  std::vector<GlmFp4Matrix> experts_fp4;  // the same, the NVFP4 release's backbone experts
  // The AutoRound hybrid's int4 g128 experts (backbone and draft) in the
  // packed core's row layout, f16 scales (docs/qwen38_autoround_int4_plan.md).
  std::vector<GlmPackedMatrix> experts_packed;
  bool packq() const { return !experts_packed.empty(); }
  float* expert_globals = nullptr;        // [E * 3] F32 on the device: 1 / weight_scale_2 per matrix
  // The layer's activation scales for W4A4 (max input_scale over the experts'
  // gate/up and over their down; 0 = not read): SGLang's flashinfer_cutlass form.
  float act_scale_w13 = 0.0f, act_scale_w2 = 0.0f;
  float* act_scales = nullptr;  // [2] F32 in the layer image: max gate/up, down input_scale
  bool nvfp4() const { return !experts_fp4.empty(); }
  int64_t local_inter = 0;                // I/W
  int64_t local_shared_inter = 0;         // S/W
  int scale_block = 128;                  // gcd(128, I/W): the experts' sliced-axis scale grid
  const GlmQuantMatrix& expert(int e, int i) const {
    return experts[static_cast<size_t>(e) * 3 + i];
  }
};

struct QwenPleResident {
  const uint16_t* key_proj = nullptr;    // BF16 [W, E/world] (packed columns)
  const uint16_t* value_proj = nullptr;  // BF16 [hidden, E/world] (packed columns)
  GlmQuantMatrix key_proj_fp8, value_proj_fp8;  // dense_weights fp8
  const uint16_t* norm_key = nullptr;    // BF16 [W]
  const uint16_t* norm_query = nullptr;  // BF16 [W]
  const uint16_t* norm_conv = nullptr;   // BF16 [W]
  const uint16_t* conv = nullptr;        // BF16 [W, k]
  float table_scale = 0.f;               // the table's per-tensor scale (host copy)
  int hash_head_begin = 0;               // this rank's first hash head
  int hash_heads = 0;                    // 16/W
  int64_t row_begin = 0;                 // the rank's global row range in the table
  int64_t rows = 0;
};

struct QwenLayerResident {
  int layer = -1;
  QwenLayerKind kind = QwenLayerKind::Gdn;
  bool has_ple = false;
  QwenGrResident attn_gr;
  QwenGrResident mlp_gr;
  QwenGdnResident gdn;  // GDN layers
  QwenQsaResident qsa;  // QSA layers (the draft layer too)
  QwenMoeResident moe;
  QwenPleResident ple;  // the PLE layer only
  size_t bytes = 0;
};

struct QwenGlobalsResident {
  const uint16_t* embed = nullptr;    // BF16 [vocab, hidden]
  const uint16_t* lm_head = nullptr;  // BF16 [lm_vocab_count, hidden]
  GlmQuantMatrix lm_head_fp8;         // dense_weights fp8
  GlmPackedMatrix lm_head_packed;     // the AutoRound hybrid's int8 g128 head (packed rows, f16 scales)
  // The opt-in draft vocabulary slice (engine.draft_vocab): the head rows
  // of the set, in the plane layout, and the set's ids (device, ascending).
  GlmPackedMatrix draft_head_packed;
  const int32_t* draft_vocab_ids = nullptr;
  int draft_vocab_count = 0;
  int lm_vocab_begin = 0;
  int lm_vocab_count = 0;
  QwenGrResident mixer;               // the final read (no inject)
  // The draft head (when the draft layer exists).
  const uint16_t* mtp_fc_embedding = nullptr;         // BF16 [hidden, hidden]
  const uint16_t* mtp_fc_hidden = nullptr;            // BF16 [hidden, hidden]
  const uint16_t* mtp_pre_fc_norm_embedding = nullptr;  // BF16 [hidden]
  const uint16_t* mtp_pre_fc_norm_hidden = nullptr;     // BF16 [W]
  QwenGrResident mtp_mixer;
  size_t bytes = 0;
};

// This rank's slice of the n-gram table: rows [row_begin, row_begin+rows)
// of the padded table, e4m3, one device allocation of its own.
// The n-gram table kept on the NVMe (2026-09-10, `engine.ngram_table =
// "mmap"`): the checkpoint shard(s) holding the table's split_ngram_parts
// row shards mapped read-only with random-access advice, never copied to
// the device. The host gathers the rows a step needs (16 per token, 160 B
// each) into pinned staging that the staged gather kernel converts; the
// page cache keeps what fits beside the model, the NVMe serves the rest
// (~90 us a page at queue depth 1, ~110 K IOPS deep, measured).
class QwenNgramTableMmap {
 public:
  struct Part {
    std::string path;      // the shard file
    uint64_t data_begin;   // the part's first byte in that file
    int64_t rows;          // rows in this part
  };
  // `parts` in shard order (part s holds rows [s * capacity, +rows)).
  QwenNgramTableMmap(std::vector<Part> parts, int64_t capacity, int head_dim);
  ~QwenNgramTableMmap();
  QwenNgramTableMmap(const QwenNgramTableMmap&) = delete;
  QwenNgramTableMmap& operator=(const QwenNgramTableMmap&) = delete;
  // The mapped bytes of global row `row`.
  const uint8_t* row(int64_t row) const;
  // dst[(t * heads_local + hl) * head_dim ..] = the row of ids[t * heads +
  // head_begin + hl], for t < n: the staging the device converts. Faults
  // the pages in parallel (a thread per 64 rows past the first 64).
  void gather(const int32_t* ids, int n, int heads, int head_begin, int heads_local,
              uint8_t* dst) const;
  int64_t capacity() const { return capacity_; }
  int head_dim() const { return head_dim_; }
  size_t mapped_bytes() const { return mapped_bytes_; }

 private:
  struct Mapping {
    int fd = -1;
    uint8_t* base = nullptr;
    size_t len = 0;
  };
  std::vector<Part> parts_;
  std::vector<const uint8_t*> part_base_;  // parts_[s]'s first row
  std::vector<Mapping> maps_;
  int64_t capacity_ = 0;
  int head_dim_ = 0;
  size_t mapped_bytes_ = 0;
};

struct QwenNgramTableResident {
  const uint8_t* rows_e4m3 = nullptr;  // [rows, head_dim] (null under the mmap'ed table)
  const QwenNgramTableMmap* mmap = nullptr;  // the mmap'ed table, else null
  int64_t row_begin = 0;
  int64_t rows = 0;
  int head_dim = 0;
  float scale = 0.f;
  size_t bytes = 0;
};


// The local TP geometry at (rank, world): every slice bound the builders
// and the views use, in one place.
struct QwenLocalGeometry {
  int world = 1, rank = 0;
  int local_key_heads = 0, local_value_heads = 0;   // GDN
  int local_heads = 0, head_begin = 0;              // QSA query heads
  int local_kv_heads = 0, kv_head_begin = 0;        // QSA kv heads
  int64_t local_inter = 0, local_shared_inter = 0;  // MoE
  int scale_block = 128;                            // gcd(128, local_inter)
  int hash_heads = 0, hash_head_begin = 0;          // PLE
  int64_t table_row_begin = 0, table_rows = 0;      // the n-gram table slice
  int lm_vocab_begin = 0, lm_vocab_count = 0;       // the lm head slice
  static QwenLocalGeometry from_config(const QwenTextConfig& cfg, int rank, int world,
                                       QwenHeadSharding head);
};

// The family behind the shared stream (loaders/resident_stream.hpp).
struct QwenLoaderFamily {
  using Config = QwenTextConfig;
  using Expected = QwenExpectedTensor;
  using LayerResident = QwenLayerResident;
  using GlobalsResident = QwenGlobalsResident;
  using Geometry = QwenLocalGeometry;
  using PresentMap = std::unordered_map<std::string, QwenTensorDesc>;
  struct Builder;  // models/qwen/loader.cpp
  static const char* who() { return "qwen loader"; }
  // The resident layout's version; the dense stack's form is part of it
  // (a BF16 image can never be restored into an FP8 world).
  static uint64_t loader_format();
  static int max_layer(const Config& c) { return c.num_hidden_layers + (c.mtp_layer() >= 0 ? 1 : 0); }
  static int main_layers(const Config& c) { return c.num_hidden_layers; }
  static std::vector<Expected> layer_table(const Config& c, int layer) {
    return qwen_expected_layer_tensors(c, layer);
  }
  static std::vector<Expected> global_table(const Config& c) { return qwen_expected_global_tensors(c); }
  static void validate_binding(const Config& c, const PresentMap& present);
  static void check_sources(const Config& c, const LoaderTensorMap& tensors);
  static bool digest_included(const Expected& e);
  static bool discard_after_pack(const Expected& e);
  static void build_globals(const Config& c, const Geometry& geo, const LoaderTensorMap& tensors,
                            LayerBump& bump, GlobalsResident& out, uint64_t& source_bytes,
                            uint64_t& verbatim_bytes, LoaderHeadSharding head);
  static size_t globals_bytes(const Config& c, int rank, int world, LoaderHeadSharding head);
  static size_t extra_resident_bytes(const Config& c, int rank, int world);
  static size_t min_staging_bytes() { return size_t{256} << 20; }  // the table's chunked copy
  static void after_restore(const Config& c, int layer, const LoaderTensorMap& tensors,
                            LayerResident& out);
  // The extra shard directory and its admission filter (loaders/resident_stream.hpp).
  static std::string extra_shard_dir();
  static bool admit_extra_tensor(const std::string& name);
};

extern template class ResidentLayerStream<QwenLoaderFamily>;

class QwenLayerStream : public ResidentLayerStream<QwenLoaderFamily> {
 public:
  QwenLayerStream(const QwenTextConfig& cfg, const std::string& checkpoint_dir, int rank = 0,
                  int world = 1, QwenResidency residency = QwenResidency::Streaming,
                  QwenHeadSharding head = QwenHeadSharding::Full, bool resident_mtp = false);
  ~QwenLayerStream() override;

  // The rank's n-gram table slice, loaded once and kept (both residencies).
  const QwenNgramTableResident& load_ngram_table();
  static size_t ngram_table_bytes(const QwenTextConfig& cfg, int rank = 0, int world = 1);
  // The process-wide table mode: true = the table stays on the
  // NVMe behind load_ngram_table()'s mapping (ngram_mmap()), rows_e4m3 null,
  // the memory plan's table bytes zero. Set before the stream is built; the
  // deployment config's engine.ngram_table ("resident" | "mmap") drives it.
  static void set_ngram_table_mmap(bool on);
  static bool ngram_table_mmap();
  // A second directory whose safetensors shards carry the n-gram table
  // (engine.ngram_table_model resolved to its snapshot): only the table's
  // tensors are admitted from it, the checkpoint's own win on a clash.
  // Empty (the default): the checkpoint's shards alone. Set before the
  // stream is built.
  static void set_ngram_table_dir(const std::string& dir);
  static const std::string& ngram_table_dir();
  // The dense stack's form (2026-09-10, engine.dense_weights): false = the
  // checkpoint's BF16 (the default); true = every dense projection (GDN
  // qkv/z/out, QSA q/k/v/o/indexer, the GR sites, the shared experts, the
  // PLE key/value, lm_head) encoded to block FP8 at load. Set before the
  // stream is built; the memory plan and the resident image key follow it.
  static void set_dense_weights_fp8(bool on);
  static bool dense_weights_fp8();
  // The opt-in fp8 prefill GEMM (2026-09-30, engine.prefill_fp8_gemm; NOT
  // bitwise): a prefill-shaped dense product under dense_weights = fp8
  // runs on the fp8 tensor cores (kernels/fp8_gemm: per-token 1 x 128
  // e4m3 activations, the 128 x 128 weight scales, fp32 promotion per
  // group) instead of dequantizing the matrix for the bf16 GEMM. Set
  // before the model is built (the activation scratch follows it).
  static void set_prefill_fp8_gemm(bool on);
  static bool prefill_fp8_gemm();
  // The chunk-ahead n-gram staging (engine.ngram_prestage, default on;
  // round 19: the next chunk's rows gathered while this one runs, bitwise).
  // Off keeps the one-channel staging. Set before the model is built.
  static void set_ngram_prestage(bool on);
  static bool ngram_prestage();
  // The draft vocabulary slice (engine.draft_vocab, 2026-09-29): a .npy of
  // token ids read once here (sorted, unique); the hybrid's loader gathers
  // those rows of the int8 head into the draft's own plane-layout matrix.
  // False with `err` set when the file cannot be read.
  static bool set_draft_vocab(const std::string& npy_path, std::string* err);
  static int draft_vocab_count();
  static const std::vector<int32_t>& draft_vocab_ids();
  // The RadixArk MTP expert format (engine.mtp_expert_format = "bf16_fused"):
  // fused BF16 gate_up_proj + down_proj instead of per-expert FP8 tensors.
  // Set before the stream is built; the memory plan and the resident image
  // key follow it.
  static void set_mtp_expert_format(bool on);
  static bool mtp_experts_bf16_fused();
  const QwenNgramTableMmap* ngram_mmap() const { return mmap_table_.get(); }

  static void set_resident_image_dir(const std::string& dir);
  static const std::string& resident_image_dir();

 protected:
  const std::string& image_dir() const override { return resident_image_dir(); }

 private:
  QwenNgramTableResident table_;
  std::unique_ptr<QwenNgramTableMmap> mmap_table_;
  void* table_device_ = nullptr;
};

}  // namespace dgpp
