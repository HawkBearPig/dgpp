#pragma once
// Per-rank blocked DSA cache pool (M3 layer phase, DESIGN §7.2/§8).
//
// Layout (all device memory, one allocation per region):
//   latent      [num_dsa_layers][max_token_slots] rows in the cache's format
//               (BF16 [kv_lora_rank], or fp8/fp4 codes — latent_format.hpp;
//               the bf16 rope tail after the payload where the model has one)
//   latent_scale [num_dsa_layers][max_token_slots]                  FP32
//               (fp8/fp4 only: one row scale per token)
//   index_k     [index_layers][max_pool_slots][index_head_dim]      FP8
//   index_scale [index_layers][max_pool_slots]                      FP32
//   tail        [index_layers][max_requests][2][kpool][dim]         BF16
//   block table [max_requests][total_blocks]                        INT32
//
// index_layers (DsaConfig::index_layers()) is every DSA layer for
// GLM-5.3-Flash; the full GLM-5.3 indexes a subset (plan D4: the "shared"
// layers attend with the last "full" layer's selection), so a layer maps
// to its index-cache ordinal through the table init() takes.
//
// The block table is the co-location pin (DESIGN §7.2): block b of request r
// holds the request's tokens [b*block_tokens, (b+1)*block_tokens) in every
// layer's latent cache AND pools [b*pools_per_block, ...) in every layer's
// index cache, through one physical block id. Prefix attachment (§8) shares
// blocks by reference; M3 only provides the allocator.
//
// Block management is host-side (a LIFO free list + mirrored table); table
// updates are small stream-ordered uploads. Acquired blocks are not scrubbed
// on release: a new owner fully rewrites the rows it will read (prefill
// writes latent rows and pool rows before any select or attention touches
// them), so release never lands on the memory hot path.
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "core/arena.hpp"
#include "engine/cache_planes.hpp"
#include "models/dsa_geometry.hpp"

namespace dgpp {

class DsaStatePool {
 public:
  DsaStatePool() = default;
  DsaStatePool(const DsaStatePool&) = delete;
  DsaStatePool& operator=(const DsaStatePool&) = delete;

  // Allocates the caches for cfg.num_dsa_layers layers, tail rings for
  // max_requests request slots, and the shared block table sized by
  // max_token_slots (which must be a multiple of block_tokens — the pool
  // capacity in tokens is shared by all requests).
  //   index_ordinal (optional): per DSA layer, its index-cache ordinal in
  //   [0, cfg.index_layers()) or -1 for a layer without one; every ordinal
  //   used exactly once. Empty = the identity (every layer indexed), which
  //   requires cfg.index_layers() == cfg.num_dsa_layers.
  void init(Arena& arena, const DsaConfig& cfg, int max_requests,
            int64_t max_token_slots, const std::vector<int>& index_ordinal = {});

  int max_requests() const { return max_requests_; }
  int64_t max_token_slots() const { return max_token_slots_; }
  int64_t total_blocks() const { return total_blocks_; }
  int64_t blocks_in_use() const;
  const DsaConfig& config() const { return cfg_; }
  const DsaGeometry& geometry() const { return geo_; }

  // Per-layer cache views (device pointers; kernels index them physically).
  //   latent(layer):      [max_token_slots] rows of latent_bytes_per_token
  //   latent_scale(layer): FP32 [max_token_slots] (nullptr for bf16)
  //   index_k(layer):     FP8  [max_pool_slots, index_head_dim]
  //   index_scale(layer): FP32 [max_pool_slots]
  //   tail(layer):        BF16 [max_requests, 2, kpool, index_head_dim]
  // The index views throw for a layer without an index cache
  // (owns_index(layer) false).
  bool owns_index(int layer) const { return index_ordinal_of(layer) >= 0; }
  int index_ordinal(int layer) const;  // -1 without one
  void* latent(int layer);
  float* latent_scale(int layer);
  void* index_k(int layer);
  float* index_scale(int layer);
  void* tail(int layer);
  const void* latent(int layer) const;
  const float* latent_scale(int layer) const;
  const void* index_k(int layer) const;
  const float* index_scale(int layer) const;
  const void* tail(int layer) const;

  // Device block table, int32 [max_requests, total_blocks]. This pointer is
  // the row-strided 2-D table kernels receive (blocks_per_request ==
  // total_blocks); row `req` is the request's logical->physical mapping.
  const int32_t* block_tables() const { return block_tables_; }

  // ---- block management (host side) ------------------------------------
  // Transactionally grows req's table to cover `tokens` tokens, acquiring
  // physical blocks and uploading the new table slice on `stream` (the
  // caller's step kernels are enqueued on the same stream afterwards).
  // Returns false without side effects when the pool cannot satisfy the
  // request — the engine's admission controller (DESIGN §9) must guarantee
  // capacity before this point.
  bool ensure_request_blocks(int req, int64_t tokens, cudaStream_t stream);

  // Releases all of req's blocks to the free list and zeroes its table row
  // (the zeroed slice is uploaded on `stream`). Cache contents are not
  // scrubbed — see the file header.
  void release_request_blocks(int req, cudaStream_t stream);

  // Blocks req currently holds (covers request_blocks()*block_tokens tokens).
  int64_t request_blocks(int req) const;

  // Blocks covering `tokens` tokens (rounds up). Public because it is the
  // admission-budget arithmetic: the scheduler reserves
  // block_count_for_tokens(prompt + max_steps) per request (M6 Stage 2b).
  int64_t block_count_for_tokens(int64_t tokens) const;

  // ---- sharing (M7, DESIGN §8; 2026-09-05) ---------------------------------
  // Blocks are REFCOUNTED: a request's table row holds one reference per
  // block, a prefix-cache entry pins its blocks with one more, and a block
  // returns to the free list when its count reaches zero. release_request_
  // blocks() only drops the request's references.
  //
  // share_blocks_into: a FRESH request row (no blocks held) takes `n`
  // physical blocks by reference as its logical blocks [0, n) — the shared
  // immutable prefix — and uploads the slice. Returns false (no side
  // effects) when n exceeds the pool. The blocks must be live (pinned or
  // held by another row): the caller's cache entry guarantees it.
  bool share_blocks_into(int req, const int32_t* blocks, int64_t n,
                         cudaStream_t stream);
  // Entry-side references without a request row.
  void pin_blocks(const int32_t* blocks, int64_t n);
  void unpin_blocks(const int32_t* blocks, int64_t n);
  // A fresh block owned by the caller (refcount 1, no row) — the copy-on-
  // attach home of a prefix's partial last block; -1 when the pool is
  // empty. Release it with unpin_blocks(&b, 1).
  int32_t acquire_pinned_block();
  // Copies every layer's cache rows of physical block `src` into `dst`
  // (latent rows, index pools and their scales), stream-ordered.
  void copy_block_contents(int32_t src, int32_t dst, cudaStream_t stream);
  // The request's logical->physical row (host mirror; request_blocks()
  // entries are meaningful).
  const int32_t* request_table_row(int req) const;
  int64_t free_blocks() const { return int64_t(free_.size()); }
  int32_t block_refcount(int32_t block) const { return refcount_[size_t(block)]; }
  // The NVMe cold tier's view (issue #26, engine/cache_planes.hpp): the
  // block's identity (its id and the generation of its contents, bumped
  // at every acquisition) and every plane a block spans.
  uint64_t block_identity(int32_t block) const;
  std::vector<CachePlane> planes() const;

  // Cold start: zero every cache, tail ring, and table row, and return all
  // blocks to the free list. One call at init or between test cases.
  void reset_all(cudaStream_t stream);

  // Per-request open (M6 Stage 2b): zero req's tail rings and return any
  // blocks it still holds to the free list. The latent/index caches are
  // deliberately not scrubbed — a new owner rewrites every row it reads
  // before any kernel reads it (the release contract in the file header),
  // so opening a request never lands on the memory hot path. Pairs with
  // the engine's per-request KDA state zeroing at session_prefill(req, ..).
  void reset_request(int req, cudaStream_t stream);

  // ---- accounting (the M3 exit criterion) -------------------------------
  // Bytes this pool allocates (each region 256-byte aligned). The static
  // form sizes a hypothetical pool without allocating — use it to check the
  // ±2% criterion at deployment scale.
  static size_t cache_bytes(const DsaConfig& cfg, int max_requests,
                            int64_t max_token_slots);
  size_t cache_bytes() const {
    return cache_bytes(cfg_, max_requests_, max_token_slots_);
  }

 private:
  DsaConfig cfg_{};
  DsaGeometry geo_{};
  int max_requests_ = 0;
  int64_t max_token_slots_ = 0;
  int64_t max_pool_slots_ = 0;
  int64_t total_blocks_ = 0;
  bool initialized_ = false;
  std::vector<int> index_ordinal_;  // per DSA layer (-1 = no index cache)
  int index_ordinal_of(int layer) const {
    return (layer < 0 || layer >= int(index_ordinal_.size())) ? -1
                                                              : index_ordinal_[size_t(layer)];
  }

  uint8_t* latent_base_ = nullptr;
  float* latent_scale_base_ = nullptr;  // fp8/fp4 rows' scales; null for bf16
  uint8_t* index_k_base_ = nullptr;
  float* index_scale_base_ = nullptr;
  uint8_t* tail_base_ = nullptr;
  int32_t* block_tables_ = nullptr;

  // Host-side allocator state (the only mutable host state; the caches
  // themselves are only touched by kernels on device).
  std::vector<int32_t> tables_host_;  // mirror of block_tables_
  std::vector<int32_t> held_;         // blocks per request
  std::vector<int32_t> free_;         // LIFO free list
  std::vector<int32_t> refcount_;     // per physical block (M7 sharing)
  std::vector<uint64_t> generation_;  // per physical block: acquisitions so far
};

}  // namespace dgpp
