#pragma once
// Main K/V and the unpooled index K share a paged block table.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "engine/cache_planes.hpp"
#include "engine/paged_blocks.hpp"
#include "models/dflash2/draft_planes.hpp"

namespace dgpp {

struct MinimaxKvPoolShape {
  int layers = 0;    // target attention layers
  int kv_heads = 0;  // the rank's kv heads
  int dim = 128;     // head_dim
  int block_tokens = 0;
  int max_requests = 0;
  int64_t token_slots = 0;  // pool capacity in tokens, a multiple of block_tokens
  int draft_layers = 0, draft_kv_heads = 0, draft_dim = 128;
};

// One layer's caches as the kernels see them.
struct MinimaxKvCache {
  uint16_t* k_cache = nullptr;  // bf16 [slots, kv_heads * dim]
  uint16_t* v_cache = nullptr;
  uint16_t* index_cache = nullptr;        // bf16 [slots, 128], replicated across TP ranks
  const int32_t* block_tables = nullptr;  // int32 [max_requests, blocks_per_request]
  int block_tokens = 0;
  int blocks_per_request = 0;
};

class MinimaxKvPool {
 public:
  MinimaxKvPool() = default;
  ~MinimaxKvPool();
  MinimaxKvPool(const MinimaxKvPool&) = delete;
  MinimaxKvPool& operator=(const MinimaxKvPool&) = delete;

  void init(const MinimaxKvPoolShape& shape);
  bool initialized() const { return initialized_; }
  static size_t cache_bytes(const MinimaxKvPoolShape& shape);

  const MinimaxKvPoolShape& shape() const { return shape_; }
  int64_t total_blocks() const { return table_.total_blocks(); }
  int64_t token_slots() const { return shape_.token_slots; }
  int64_t blocks_in_use() const { return table_.blocks_in_use(); }
  int64_t free_blocks() const { return table_.free_blocks(); }
  int64_t block_count_for_tokens(int64_t tokens) const {
    return table_.block_count_for_tokens(tokens);
  }
  const PagedBlockTable& blocks() const { return table_; }

  MinimaxKvCache view(int layer) const;
  DraftKvPlanes::View draft_view(int layer) const { return draft_.view(layer); }

  bool ensure_request_blocks(int req, int64_t tokens, cudaStream_t stream) {
    return table_.ensure_request_blocks(req, tokens, stream);
  }
  void release_request_blocks(int req, cudaStream_t stream) {
    table_.release_request_blocks(req, stream);
  }
  int64_t request_blocks(int req) const { return table_.request_blocks(req); }
  const int32_t* request_table_row(int req) const { return table_.request_table_row(req); }
  void reset_all(cudaStream_t stream);

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
  // plane a block spans (each layer's K rows and V rows).
  uint64_t block_identity(int32_t block) const { return table_.block_identity(block); }
  std::vector<CachePlane> planes() const;

 private:
  MinimaxKvPoolShape shape_;
  bool initialized_ = false;
  PagedBlockTable table_;
  DraftKvPlanes draft_;
  uint16_t* k_base_ = nullptr;  // [layers][token_slots][kv_heads * dim]
  uint16_t* v_base_ = nullptr;
  uint16_t* index_base_ = nullptr;
  size_t kv_row_elems() const { return static_cast<size_t>(shape_.kv_heads) * shape_.dim; }
  size_t layer_kv_elems() const { return static_cast<size_t>(shape_.token_slots) * kv_row_elems(); }
};

}  // namespace dgpp
