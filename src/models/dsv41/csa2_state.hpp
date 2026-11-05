#pragma once
// The CSA2 caches of one rank (2026-09-13, docs/deepseek_v41_flash_plan.md
// D7): the compressed main KV (fp4_block rows) and the index keys (planar
// e4m3 + a row scale) of every kv-source layer, in the paged pool behind
// one PagedBlockTable of 128-token blocks — a cache at ratio r holds
// 128 / r entries per block, so a block is 128 TOKENS on every plane and
// the prefix cache shares by reference as for DSA; the per-request window
// rings of every attention layer (ring_slots rows of the fp8_block form;
// slot = pos % ring_slots, positional: a rejected draft's slot is never
// read by a later query, so the rings are not snapshot state); and the
// compressor tails of the ratio-2 kv sources (fp32 [2, 512] per request:
// the pending even token's kv and gate score — the one state family
// with per-row snapshots).
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "engine/cache_planes.hpp"
#include "engine/paged_blocks.hpp"
#include "kernels/csa2.hpp"
#include "kernels/latent_format.hpp"

namespace dgpp {

struct Csa2PoolShape {
  int layers = 0;                // attention layers with a window ring (the draft stages included)
  std::vector<int> cache_ratio;  // per cache ordinal (a kv source), its compress ratio (1 or 2)
  int tail_ordinals = 0;         // ratio-2 kv sources (compressor tails)
  int max_requests = 0;
  int64_t token_slots = 0;       // pool capacity in tokens, a multiple of block_tokens
  int block_tokens = 128;
  int ring_slots = 160;
};

class Csa2StatePool {
 public:
  Csa2StatePool() = default;
  ~Csa2StatePool();
  Csa2StatePool(const Csa2StatePool&) = delete;
  Csa2StatePool& operator=(const Csa2StatePool&) = delete;

  void init(const Csa2PoolShape& shape);
  bool initialized() const { return initialized_; }
  static size_t cache_bytes(const Csa2PoolShape& shape);
  static constexpr LatentFormat kMainFormat = LatentFormat::kFp4Block;
  static constexpr LatentFormat kRingFormat = LatentFormat::kFp8Block;

  const Csa2PoolShape& shape() const { return shape_; }
  int caches() const { return static_cast<int>(shape_.cache_ratio.size()); }
  int cache_ratio(int ord) const { return shape_.cache_ratio[static_cast<size_t>(ord)]; }
  int entries_per_block(int ord) const { return shape_.block_tokens / cache_ratio(ord); }
  int64_t entry_slots(int ord) const { return table_.total_blocks() * entries_per_block(ord); }
  size_t main_row_bytes() const { return latent_row_bytes(kMainFormat, kCsa2Latent); }
  size_t ring_row_bytes() const { return latent_row_bytes(kRingFormat, kCsa2Latent); }
  size_t ring_bytes_per_request() const { return static_cast<size_t>(shape_.ring_slots) * ring_row_bytes(); }
  int64_t total_blocks() const { return table_.total_blocks(); }
  int64_t token_slots() const { return shape_.token_slots; }
  int64_t blocks_in_use() const { return table_.blocks_in_use(); }
  int64_t free_blocks() const { return table_.free_blocks(); }
  int64_t block_count_for_tokens(int64_t tokens) const { return table_.block_count_for_tokens(tokens); }
  const PagedBlockTable& blocks() const { return table_; }
  const int32_t* block_tables() const { return table_.device_tables(); }

  // The planes (device pointers; the kernels index them physically).
  uint8_t* main(int ord) const;
  uint8_t* index_k(int ord) const;
  float* index_scale(int ord) const;
  uint8_t* ring(int layer) const;         // [max_requests][ring_slots] rows
  const int32_t* ring_table() const { return ring_table_; }  // the identity [max_requests]
  float* tails(int tail_ord) const;       // [max_requests][2][512]
  size_t tail_bytes_per_request() const { return 2 * kCsa2Latent * sizeof(float); }

  // ---- block management (the shared table's protocol) --------------------
  bool ensure_request_blocks(int req, int64_t tokens, cudaStream_t stream) {
    return table_.ensure_request_blocks(req, tokens, stream);
  }
  void release_request_blocks(int req, cudaStream_t stream) { table_.release_request_blocks(req, stream); }
  int64_t request_blocks(int req) const { return table_.request_blocks(req); }
  const int32_t* request_table_row(int req) const { return table_.request_table_row(req); }
  // Per-request open: zero req's tails and rings, release its blocks.
  void reset_request(int req, cudaStream_t stream);
  // Cold start: every plane, ring, tail and table row zero; all blocks free.
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
  // The NVMe cold tier's view (issue #26): the block's identity and every
  // plane a block spans (each cache's main rows, index keys and scales).
  uint64_t block_identity(int32_t block) const { return table_.block_identity(block); }
  std::vector<CachePlane> planes() const;

 private:
  Csa2PoolShape shape_;
  bool initialized_ = false;
  PagedBlockTable table_;
  std::vector<uint8_t*> main_;         // per cache ordinal
  std::vector<uint8_t*> index_k_;
  std::vector<float*> index_scale_;
  uint8_t* ring_base_ = nullptr;       // [layers][max_requests][ring_slots] rows
  int32_t* ring_table_ = nullptr;
  float* tail_base_ = nullptr;         // [tail_ordinals][max_requests][2][512]
  void check_ord(int ord, const char* what) const;
};

}  // namespace dgpp
