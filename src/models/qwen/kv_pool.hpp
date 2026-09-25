#pragma once
// QwenKvPool: the QSA layers' paged caches for max_requests request slots
// (Q6, 2026-09-09; docs/qwen38_flash_next_plan.md D5, the DSA pool's design
// in models/dsa_state.hpp). Per attention layer: K and V bf16 rows for the
// rank's kv heads, the compressed index keys (bf16, one per pool of kpool
// tokens) co-located with the token blocks, and a per-request ring of the
// pending raw keys. one block table int32 [max_requests, total_blocks] is
// shared by every layer: block b of request r holds the request's tokens
// [b*block_tokens, (b+1)*block_tokens) in every layer's K/V cache AND pools
// [b*pools_per_block, ...) in every layer's index cache, through one
// physical block id (blocks_per_request == total_blocks: one request may
// take the whole pool). The table, its refcounts and the sharing/pinning
// protocol are the shared PagedBlockTable (engine/paged_blocks.hpp,
// extracted 2026-09-09); this pool owns the cache planes. The ring is the
// one cache a fresh request reads before writing (the decode update's pool
// assembly), so reset_request zeroes it.
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "engine/cache_planes.hpp"
#include "engine/paged_blocks.hpp"
#include "models/qwen/layers.hpp"

namespace dgpp {

struct QwenKvPoolShape {
  int layers = 0;          // QSA layers served
  int kv_heads = 0;        // the rank's kv heads
  int dim = 0;             // head_dim
  int idx_dim = 0;         // indexer head dim
  int kpool = 0;           // indexer_compress_ratio
  int block_tokens = 0;    // a multiple of kpool
  int max_requests = 0;
  int64_t token_slots = 0; // pool capacity in tokens, a multiple of block_tokens
};

class QwenKvPool {
 public:
  QwenKvPool() = default;
  ~QwenKvPool();
  QwenKvPool(const QwenKvPool&) = delete;
  QwenKvPool& operator=(const QwenKvPool&) = delete;

  void init(const QwenKvPoolShape& shape);
  bool initialized() const { return initialized_; }
  // The device bytes init allocates for a shape (the memory plan).
  static size_t cache_bytes(const QwenKvPoolShape& shape);

  const QwenKvPoolShape& shape() const { return shape_; }
  int64_t total_blocks() const { return table_.total_blocks(); }
  int64_t token_slots() const { return shape_.token_slots; }
  int64_t pool_slots() const { return shape_.token_slots / shape_.kpool; }
  int pools_per_block() const { return shape_.block_tokens / shape_.kpool; }
  int64_t blocks_in_use() const { return table_.blocks_in_use(); }
  int64_t free_blocks() const { return table_.free_blocks(); }
  int64_t block_count_for_tokens(int64_t tokens) const { return table_.block_count_for_tokens(tokens); }
  const PagedBlockTable& blocks() const { return table_; }

  // The kernels' view of one layer's caches (the shared table inside).
  QwenQsaCache view(int layer) const;
  uint16_t* ring(int layer, int req) const;
  size_t ring_bytes_per_request() const;

  // ---- block management (the shared table's protocol) --------------------
  bool ensure_request_blocks(int req, int64_t tokens, cudaStream_t stream) {
    return table_.ensure_request_blocks(req, tokens, stream);
  }
  void release_request_blocks(int req, cudaStream_t stream) {
    table_.release_request_blocks(req, stream);
  }
  int64_t request_blocks(int req) const { return table_.request_blocks(req); }
  const int32_t* request_table_row(int req) const { return table_.request_table_row(req); }
  // Per-request open: zero req's rings in every layer, release its blocks.
  void reset_request(int req, cudaStream_t stream);
  // Cold start: zero every cache, ring and table row; all blocks free.
  void reset_all(cudaStream_t stream);

  // ---- sharing (the prefix cache) --------------------------------------
  bool share_blocks_into(int req, const int32_t* blocks, int64_t n, cudaStream_t stream) {
    return table_.share_blocks_into(req, blocks, n, stream);
  }
  void pin_blocks(const int32_t* blocks, int64_t n) { table_.pin_blocks(blocks, n); }
  void unpin_blocks(const int32_t* blocks, int64_t n) { table_.unpin_blocks(blocks, n); }
  int32_t acquire_pinned_block() { return table_.acquire_pinned_block(); }
  // Every layer's rows of physical block `src` into `dst`, stream-ordered.
  void copy_block_contents(int32_t src, int32_t dst, cudaStream_t stream);
  int32_t block_refcount(int32_t block) const { return table_.block_refcount(block); }
  // The NVMe cold tier's view (issue #26): the block's identity and every
  // plane a block spans (each layer's K rows, V rows and index keys).
  uint64_t block_identity(int32_t block) const { return table_.block_identity(block); }
  std::vector<CachePlane> planes() const;

 private:
  QwenKvPoolShape shape_;
  bool initialized_ = false;
  PagedBlockTable table_;
  uint16_t* k_base_ = nullptr;      // [layers][token_slots][kv_heads * dim]
  uint16_t* v_base_ = nullptr;
  uint16_t* idx_base_ = nullptr;    // [layers][pool_slots][idx_dim]
  uint16_t* ring_base_ = nullptr;   // [layers][max_requests][kpool][idx_dim]
  size_t kv_row_elems() const { return static_cast<size_t>(shape_.kv_heads) * shape_.dim; }
  size_t layer_kv_elems() const { return static_cast<size_t>(shape_.token_slots) * kv_row_elems(); }
  size_t layer_idx_elems() const { return static_cast<size_t>(pool_slots()) * shape_.idx_dim; }
  size_t layer_ring_elems() const {
    return static_cast<size_t>(shape_.max_requests) * shape_.kpool * shape_.idx_dim;
  }
  void check_req(int req, const char* what) const;
};

}  // namespace dgpp
