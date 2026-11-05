#pragma once
// DoryKvPool: the Dory attention layers' paged K/V caches for max_requests
// request slots (2026-10-18, docs/dory_plan.md §1.6/§2, D2). The pool is the
// engine's (the GLM-4.7 / MiMo shape) with one difference: the recurrent
// block's attention layers run once per loop, so a layer carries ONE PLANE
// PER LOOP under `per_loop` (the config's `kv_repeats`) and a single plane
// under `last_loop`, overwritten by each loop. Every plane holds the FULL
// history — the sliding layers' 1024-token window is applied at read time,
// the MiMo precedent, so one block list restores a prefix on every layer.
//
// `repeats` is indexed by PHYSICAL layer (0 on the MLP layers), so the walk
// addresses plane_of(layer, loop); the planes are laid out back to back at
// their own offsets, every one of them the same kv_heads * dim row (all 36
// attention layers share the 4/256 geometry).
//
// The block table, its refcounts and the sharing/pinning protocol are the
// shared PagedBlockTable (engine/paged_blocks.hpp); this pool owns the
// planes. Acquired blocks are not scrubbed on release: the prefill's append
// writes every row before any attention reads it.
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "engine/paged_blocks.hpp"
#include "engine/cache_planes.hpp"

namespace dgpp {

struct DoryKvPoolShape {
  std::vector<int> repeats;  // per physical layer: 0 (MLP), or kv_repeats (>= 1)
  int kv_heads = 4;
  int dim = 256;             // head_dim
  int block_tokens = 0;
  int max_requests = 0;
  int64_t token_slots = 0;   // pool capacity in tokens, a multiple of block_tokens

  int layers() const { return static_cast<int>(repeats.size()); }
  int planes() const {
    int n = 0;
    for (int r : repeats) n += r;
    return n;
  }
};

// One plane as the kernels see it.
struct DoryKvCache {
  uint16_t* k_cache = nullptr;             // bf16 [slots, kv_heads * dim]
  uint16_t* v_cache = nullptr;
  const int32_t* block_tables = nullptr;   // int32 [max_requests, blocks_per_request]
  int block_tokens = 0;
  int blocks_per_request = 0;
  int kv_heads = 0;
  int dim = 0;
};

class DoryKvPool {
 public:
  DoryKvPool() = default;
  ~DoryKvPool();
  DoryKvPool(const DoryKvPool&) = delete;
  DoryKvPool& operator=(const DoryKvPool&) = delete;

  void init(const DoryKvPoolShape& shape);
  bool initialized() const { return initialized_; }
  static size_t cache_bytes(const DoryKvPoolShape& shape);

  const DoryKvPoolShape& shape() const { return shape_; }
  int64_t total_blocks() const { return table_.total_blocks(); }
  int64_t token_slots() const { return shape_.token_slots; }
  int64_t blocks_in_use() const { return table_.blocks_in_use(); }
  int64_t free_blocks() const { return table_.free_blocks(); }
  int64_t block_count_for_tokens(int64_t tokens) const { return table_.block_count_for_tokens(tokens); }
  const PagedBlockTable& blocks() const { return table_; }

  // The global plane index of (layer, loop); loop is clamped to the layer's
  // repeat count (the `last_loop` policy collapses every loop onto one
  // plane, so the walk may pass a loop index freely).
  int plane_of(int layer, int loop) const;
  DoryKvCache view(int layer, int loop) const;
  DoryKvCache view_plane(int plane) const;

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
  // Every plane's rows of physical block `src` into `dst`, stream-ordered.
  void copy_block_contents(int32_t src, int32_t dst, cudaStream_t stream);
  int32_t block_refcount(int32_t block) const { return table_.block_refcount(block); }
  uint64_t block_identity(int32_t block) const { return table_.block_identity(block); }
  std::vector<CachePlane> planes() const;

 private:
  DoryKvPoolShape shape_;
  bool initialized_ = false;
  PagedBlockTable table_;
  uint16_t* k_base_ = nullptr;  // [planes][token_slots][kv_heads * dim]
  uint16_t* v_base_ = nullptr;
  std::vector<size_t> k_off_;   // [layers + 1] plane-index base per physical layer
  size_t kv_row_elems() const { return static_cast<size_t>(shape_.kv_heads) * shape_.dim; }
  size_t plane_kv_elems() const { return static_cast<size_t>(shape_.token_slots) * kv_row_elems(); }
};

}  // namespace dgpp
