#pragma once
// The attention caches of one DeepSeek-V4-Flash rank (2026-10-01): every
// compressing layer's own compressed KV (bf16 rows of 512 — the reference's
// quantize-dequantized values, kernels/dsv4_attn.hpp) and, on the ratio-4
// layers, its index keys (planar e4m3 + a row scale), in the paged pool
// behind one PagedBlockTable of 128-token blocks — a cache at ratio r
// holds 128 / r entries per block, so a block is 128 TOKENS on every plane
// and the prefix cache shares by reference; the per-request window rings
// of every attention layer (ring_slots bf16 rows; slot = pos %
// ring_slots); and the compressors' positional rings (fp32 kv and score
// per token). Every ring is positional: a rejected speculative row's slot
// is rewritten before any accepted query or group reads it, so no state
// family needs a per-row rollback snapshot.
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "engine/cache_planes.hpp"
#include "engine/paged_blocks.hpp"
#include "kernels/dsv4_attn.hpp"

namespace dgpp {

struct Dsv4PoolShape {
  int layers = 0;                // attention layers with a window ring (the draft stages included)
  std::vector<int> cache_ratio;  // per cache ordinal (a compressing layer), its ratio (4 or 128)
  int max_requests = 0;
  int64_t token_slots = 0;       // pool capacity in tokens, a multiple of block_tokens
  int block_tokens = 128;
  int ring_slots = 160;
  // > 0: the ratio-128 compressors pool lazily (kernels/dsv4_attn.hpp, the
  // lazy form) — their rings hold the open group's layer inputs, bf16
  // [slots][lazy_hidden], instead of the per-token projections. 0: the
  // per-token rings everywhere.
  int lazy_hidden = 0;
};

class Dsv4StatePool {
 public:
  Dsv4StatePool() = default;
  ~Dsv4StatePool();
  Dsv4StatePool(const Dsv4StatePool&) = delete;
  Dsv4StatePool& operator=(const Dsv4StatePool&) = delete;

  void init(const Dsv4PoolShape& shape);
  bool initialized() const { return initialized_; }
  static size_t cache_bytes(const Dsv4PoolShape& shape);
  // The compressor geometries (the reference `Compressor`): ratio 4 pools
  // overlapping groups through projections twice as wide.
  static constexpr int kSpecSlack = 8;  // speculative rows a ring absorbs past an entry's slots
  static Dsv4CompGeom main_geom(int ratio) {
    const bool ov = ratio == 4;
    return Dsv4CompGeom{ratio, kDsv4Latent, (ov ? 2 : 1) * kDsv4Latent, ov, (ov ? 2 : 1) * ratio + kSpecSlack};
  }
  static Dsv4CompGeom index_geom(int ratio) {
    return Dsv4CompGeom{ratio, kDsv4IndexDim, 2 * kDsv4IndexDim, true, 2 * ratio + kSpecSlack};
  }

  const Dsv4PoolShape& shape() const { return shape_; }
  int caches() const { return static_cast<int>(shape_.cache_ratio.size()); }
  int cache_ratio(int ord) const { return shape_.cache_ratio[static_cast<size_t>(ord)]; }
  bool indexed(int ord) const { return cache_ratio(ord) == 4; }
  int entries_per_block(int ord) const { return shape_.block_tokens / cache_ratio(ord); }
  int64_t entry_slots(int ord) const { return table_.total_blocks() * entries_per_block(ord); }
  static constexpr size_t kRowBytes = static_cast<size_t>(kDsv4Latent) * 2;
  size_t ring_bytes_per_request() const { return static_cast<size_t>(shape_.ring_slots) * kRowBytes; }
  int64_t total_blocks() const { return table_.total_blocks(); }
  int64_t token_slots() const { return shape_.token_slots; }
  int64_t blocks_in_use() const { return table_.blocks_in_use(); }
  int64_t free_blocks() const { return table_.free_blocks(); }
  int64_t block_count_for_tokens(int64_t tokens) const { return table_.block_count_for_tokens(tokens); }
  const PagedBlockTable& blocks() const { return table_; }
  const int32_t* block_tables() const { return table_.device_tables(); }

  // The planes (device pointers; the kernels index them physically).
  uint8_t* main(int ord) const;           // bf16 [entry_slots][512]
  uint8_t* index_k(int ord) const;        // e4m3 [entry_slots][128] (ratio 4)
  float* index_scale(int ord) const;      // fp32 [entry_slots] (ratio 4)
  uint8_t* ring(int layer) const;         // [max_requests][ring_slots] rows
  const int32_t* ring_table() const { return ring_table_; }  // the identity [max_requests]
  // The same rings as a table of max_requests blocks per request (row r =
  // {r, ...}): the tensor-core listed attention bounds a block id by the
  // table's blocks per request, so the one-block form above reads as an
  // anomaly for every request past the first.
  const int32_t* ring_table_wide() const { return ring_table_wide_; }
  int ring_table_wide_blocks() const { return shape_.max_requests; }
  // A lazily pooled cache (ratio 128 under shape.lazy_hidden): u_ring holds
  // its open group's layer inputs and it has no comp_ring.
  bool lazy(int ord) const { return shape_.lazy_hidden > 0 && cache_ratio(ord) == 128; }
  uint16_t* u_ring(int ord) const;        // bf16 [max_requests][main_geom(128).slots][lazy_hidden]
  float* comp_ring(int ord) const;        // fp32 [max_requests] x main_geom(ratio).ring_elems()
  float* index_comp_ring(int ord) const;  // fp32 [max_requests] x index_geom(ratio).ring_elems() (ratio 4)
  size_t u_ring_elems_per_request() const {
    return static_cast<size_t>(main_geom(128).slots) * static_cast<size_t>(shape_.lazy_hidden);
  }
  size_t comp_ring_bytes_per_request(int ord) const { return main_geom(cache_ratio(ord)).ring_elems() * sizeof(float); }
  size_t index_comp_ring_bytes_per_request(int ord) const {
    return index_geom(cache_ratio(ord)).ring_elems() * sizeof(float);
  }

  // ---- block management (the shared table's protocol) --------------------
  bool ensure_request_blocks(int req, int64_t tokens, cudaStream_t stream) {
    return table_.ensure_request_blocks(req, tokens, stream);
  }
  void release_request_blocks(int req, cudaStream_t stream) { table_.release_request_blocks(req, stream); }
  int64_t request_blocks(int req) const { return table_.request_blocks(req); }
  const int32_t* request_table_row(int req) const { return table_.request_table_row(req); }
  // Per-request open: zero req's rings, release its blocks.
  void reset_request(int req, cudaStream_t stream);
  // Cold start: every plane, ring and table row zero; all blocks free.
  void reset_all(cudaStream_t stream);
  bool share_blocks_into(int req, const int32_t* blocks, int64_t n, cudaStream_t stream) {
    return table_.share_blocks_into(req, blocks, n, stream);
  }
  void pin_blocks(const int32_t* blocks, int64_t n) { table_.pin_blocks(blocks, n); }
  void unpin_blocks(const int32_t* blocks, int64_t n) { table_.unpin_blocks(blocks, n); }
  int32_t acquire_pinned_block() { return table_.acquire_pinned_block(); }
  // Every cache's rows of physical block `src` into `dst`, stream-ordered.
  void copy_block_contents(int32_t src, int32_t dst, cudaStream_t stream);
  int32_t block_refcount(int32_t block) const { return table_.block_refcount(block); }
  // The NVMe cold tier's view (issue #26, engine/cache_planes.hpp): the
  // block's identity (its id and the generation of its contents) and every
  // plane a block spans (each cache's main rows and, where indexed, its
  // index keys and scales — the regions copy_block_contents walks).
  uint64_t block_identity(int32_t block) const { return table_.block_identity(block); }
  std::vector<CachePlane> planes() const;

 private:
  Dsv4PoolShape shape_;
  bool initialized_ = false;
  PagedBlockTable table_;
  std::vector<uint8_t*> main_;         // per cache ordinal
  std::vector<uint8_t*> index_k_;      // per cache ordinal (null: not indexed)
  std::vector<float*> index_scale_;
  std::vector<float*> comp_ring_;      // per cache ordinal (null: lazy)
  std::vector<uint16_t*> u_ring_;      // per cache ordinal (null: not lazy)
  std::vector<float*> index_comp_ring_;
  uint8_t* ring_base_ = nullptr;       // [layers][max_requests][ring_slots] rows
  int32_t* ring_table_ = nullptr;
  int32_t* ring_table_wide_ = nullptr;
  void check_ord(int ord, const char* what) const;
};

}  // namespace dgpp
