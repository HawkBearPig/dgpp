#pragma once
// The Engram layer of DeepSeek-V4.1-Flash (2026-09-13,
// docs/deepseek_v41_flash_plan.md §1.6, D5): the hashed n-gram memory
// whose tables stay mmap'ed on the NVMe (models/dsv41/loader.hpp) and
// whose rows a HOST NODE gathers while the walk runs its first layer —
// the Qwen n-gram layer's pattern (models/qwen/layers.hpp): stage() at
// the walk's start runs the hash kernel for every Engram layer into
// pinned ids and forks one host callback off the stream that gathers
// both tables' rows into pinned staging; embed(i) at layer i's turn joins
// and converts that layer's rows to bf16; project() runs this rank's
// K-slice of wkv (the fold is the model's); gate() updates the streams.
// Works eagerly and under capture alike (the fork / join are graph edges,
// the gather a host node), and every path stages from the device's own
// tokens and context — nothing mirrors them on the host.
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/dsv41_engram.hpp"
#include "kernels/gemm.hpp"
#include "models/dsv41/config.hpp"
#include "models/dsv41/engram_tables.hpp"
#include "models/dsv41/loader.hpp"
#include "models/exl3_matrix.hpp"
#include "models/quant_matrix.hpp"

namespace dgpp {

struct Dsv41EngramLayerWeights {
  GlmQuantMatrix wkv;                   // fp8 [(hc + 1) * hidden, rows_local * head_dim] (this rank's K columns)
  Exl3Matrix wkv_x;                     // the EXL3 pack's (the same K columns); set instead of wkv
  const uint16_t* q_weight = nullptr;   // bf16 [hc, hidden]
  const uint16_t* k_weight = nullptr;   // bf16 [hc, hidden]
  int table_index = -1;                 // the layer's ordinal among the Engram layers
};

class Dsv41EngramLayer {
 public:
  // `tables`: the rank's mappings (Dsv41LayerStream::load_engram_tables);
  // `sidecar`: the hash constants, uploaded once here.
  Dsv41EngramLayer(const Dsv41TextConfig& cfg, const Dsv41EngramTables& tables, const Dsv41EngramSidecar& sidecar,
                   int max_tokens, int max_requests);
  ~Dsv41EngramLayer();
  // The wkv projection's decode form (Csa2Config::dense_mma): set by the model.
  void set_dense_mma(bool on) { dense_mma_ = on; }
  Dsv41EngramLayer(const Dsv41EngramLayer&) = delete;
  Dsv41EngramLayer& operator=(const Dsv41EngramLayer&) = delete;

  // The hash of every Engram layer's rows for this walk, then the host
  // gather forked off `stream`. Rows in span order (kernels/dsv41_engram.hpp);
  // ctx: the requests' contexts (device int32 [max_requests, 4]).
  // `host_ids` / `pos0` / `req` (a scalar prefill chunk's host ids, first
  // position and slot) let stage() recognize the chunk a prestage() already
  // gathered and skip its own hash and gather.
  void stage(const int64_t* tokens, int rows, const int32_t* req_ids, const int64_t* pos,
             const int32_t* req_spans, int num_requests, const int32_t* ctx, cudaStream_t stream,
             const int64_t* host_ids = nullptr, int64_t pos0 = -1, int req = -1);
  // The chunk-ahead staging (2026-10-18; Qwen's PLE form, engine.ngram_prestage):
  // the NEXT chunk's rows hashed from its host ids (uploaded here) behind
  // the `n_prefix` tokens that precede it (the current chunk's last
  // max_ngram - 1, so every row's context is in the span), and gathered on
  // the side stream while this chunk's layers run — the 32K prefill's
  // profile had 110–160 ms of the gather exposed per 4,096-token chunk at
  // the first Engram layer, 1.4 s of the walk. The next stage() with the
  // matching (host_ids, rows, pos0, req, first and last token) takes the
  // prestaged rows; anything else drops them. n_prefix must be
  // max_ngram - 1 (a shorter current chunk stages nothing ahead).
  void prestage(const int64_t* host_ids, int rows, int64_t pos0, int req, const int64_t* prefix, int n_prefix,
                cudaStream_t stream);
  static void set_prestage(bool on);
  static bool prestage_enabled();
  uint64_t debug_prestage_publishes() const { return pre_publishes_; }
  uint64_t debug_prestage_claims() const { return pre_claims_; }
  // The context after every row into ctx_rows (per-row snapshots) and the
  // last real row's in place — enqueue after stage() (the hash reads the
  // incoming context).
  void context_rows(const int64_t* tokens, int rows, const int32_t* req_ids, const int64_t* pos,
                    const int32_t* req_spans, int num_requests, int32_t* ctx, int32_t* ctx_rows,
                    cudaStream_t stream);
  // Joins the gather and converts layer `table_index`'s staged rows into
  // e (bf16 [rows, rows_local * head_dim]); any layer, any number of
  // times, until the next stage().
  void embed(int table_index, int rows, cudaStream_t stream);
  // kv_dst [rows, (hc + 1) * hidden] = e x wkv^T (this rank's partial;
  // null: the internal kv_partial()).
  // `r0`: the first row (the overlap walk projects a row block at a time:
  // e_ rows [r0, r0 + rows) into kv rows [r0, r0 + rows)).
  void project(const Dsv41EngramLayerWeights& w, int rows, uint16_t* kv_dst, cudaStream_t stream,
               bool prefill_tile = false, int r0 = 0);
  // The stream update on the (folded) kv: x [rows, hc, hidden] in place.
  void gate(const Dsv41EngramLayerWeights& w, uint16_t* x, const uint16_t* kv, int rows, cudaStream_t stream);
  // A staging that failed on the host surfaces here (and at the next stage()).
  void check_staged() const;

  uint16_t* kv_partial() const { return kv_; }
  const uint16_t* embedded() const { return e_; }
  const int32_t* ids() const { return h_ids_; }  // pinned [max_tokens, layers, ngrams * heads]
  const Dsv41EngramHash& hash() const { return hash_; }
  int rows_local() const { return tables_.rows_local(); }
  int width_local() const { return tables_.rows_local() * cfg_.engram_head_dim; }
  // The pinned bytes (the memory plan): the ids and the staged rows.
  static size_t staging_bytes(const Dsv41TextConfig& cfg, int world, int max_tokens);
  static size_t device_bytes(const Dsv41TextConfig& cfg, int world, int max_tokens);

 private:
  bool dense_mma_ = true;
  // The EXL3 projection's workspace (kernels/exl3_gemv.hpp), allocated for an EXL3 pack.
  void* exl3_ws_ = nullptr;
  size_t exl3_ws_bytes_ = 0;
  struct StageArgs;
  static void stage_callback(void* user);

  Dsv41TextConfig cfg_;
  const Dsv41EngramTables& tables_;
  int max_tokens_ = 0;
  int layers_ = 0;
  int entries_ = 0;         // (max_ngram - 1) * heads: a layer's row of ids
  int staged_rows_ = 0;
  int eager_slot_ = 0;
  std::vector<std::unique_ptr<StageArgs>> stage_args_;
  cudaStream_t side_ = nullptr;
  cudaEvent_t fork_ = nullptr, join_ = nullptr;
  // Per table: the host's published count of stagings (pinned), the
  // device's consumed count, the wait kernel's late flag.
  uint64_t* h_ready_ = nullptr;
  uint64_t* d_ready_ = nullptr;
  uint32_t* h_late_ = nullptr;
  std::vector<uint64_t> ready_count_;
  // The hash constants on the device.
  int32_t* d_token_map_ = nullptr;
  int64_t* d_mult_ = nullptr;
  int64_t* d_primes_ = nullptr;
  int64_t* d_offsets_ = nullptr;
  Dsv41EngramHash hash_{};
  int32_t* h_ids_ = nullptr;      // pinned, device-mapped [max_tokens, layers, entries]
  uint8_t* staged_ = nullptr;     // pinned [layers][max_tokens, rows_local, row_bytes]
  size_t staged_layer_bytes_ = 0;
  // The prestage channel (prestage()): the next chunk's tokens behind their
  // prefix, their ids and staged rows, the gather's join, the chunk's key.
  std::unique_ptr<StageArgs> pre_args_;
  int64_t* h_pre_tokens_ = nullptr;  // pinned [max_tokens + 4]
  int64_t* h_pre_pos_ = nullptr;     // pinned [max_tokens + 4]
  int32_t* h_pre_meta_ = nullptr;    // pinned [2]: the one span
  int64_t* d_pre_tokens_ = nullptr;
  int64_t* d_pre_pos_ = nullptr;
  int32_t* d_pre_req_ = nullptr;     // zeros: the private context slot
  int32_t* d_pre_spans_ = nullptr;
  int32_t* d_pre_ctx_ = nullptr;     // [4] pad_class: the prefix rows hash from nothing
  int32_t* pre_ids_ = nullptr;       // pinned, device-mapped [max_tokens + 4, layers, entries]
  uint8_t* pre_staged_ = nullptr;    // pinned [layers][max_tokens, rows_local, row_bytes]
  cudaEvent_t pre_fork_ = nullptr, pre_join_ = nullptr;
  const int64_t* pre_host_ids_ = nullptr;
  int pre_rows_ = 0;
  int64_t pre_pos0_ = -1;
  int pre_req_ = -1;
  int64_t pre_first_ = 0, pre_last_ = 0;
  bool pre_pending_ = false;  // a prestage published, not yet claimed
  bool use_pre_ = false;      // the pending stage reads the prestage channel
  uint64_t pre_publishes_ = 0, pre_claims_ = 0;
  uint16_t* e_ = nullptr;         // device [max_tokens, rows_local * head_dim]
  uint16_t* kv_ = nullptr;        // device [max_tokens, (hc + 1) * hidden]
};

}  // namespace dgpp
