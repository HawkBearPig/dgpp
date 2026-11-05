#pragma once
// MimoKvPool: the MiMo-V2.6-Flash attention layers' paged K/V caches for
// max_requests request slots (2026-09-22, docs/mimo_v26_flash_plan.md D4):
// per layer, K bf16 rows [slots, kv_heads_l * 192] and V bf16 rows [slots,
// kv_heads_l * 128] for the rank's kv heads of that layer — the global
// layers hold num_key_value_heads / W, the sliding-window layers (and the
// draft) swa_num_key_value_heads / W, so the planes are laid out per layer
// at their own widths. One block table serves every layer (the shared
// PagedBlockTable, engine/paged_blocks.hpp: refcounts, sharing, pinning);
// this pool owns the planes. The sliding-window layers keep the full
// history in the paged cache like the global ones (the window is applied
// at read time): the same block list restores a prefix for every layer.
// Acquired blocks are not scrubbed on release: the prefill's append writes
// every row before any attention reads it.
//
// Two formats (engine.kv_dtype, the GLM latent caches' knob): bf16 rows,
// or the fp8 row form of kernels/latent_format.hpp applied per (token, kv
// head) — e4m3 codes with one fp32 scale (absmax / 448) per head row, the
// K and V heads each their own — 328 bytes per kv head per token against
// 640 (2026-09-22, the two-node recipe's context lever).
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "engine/cache_planes.hpp"
#include "engine/paged_blocks.hpp"
#include "kernels/latent_format.hpp"

namespace dgpp {

struct MimoKvPoolShape {
  std::vector<int> kv_heads;  // per attention layer served (the draft's last)
  int k_dim = 192;
  int v_dim = 128;
  int block_tokens = 0;
  int max_requests = 0;
  int64_t token_slots = 0;    // pool capacity in tokens, a multiple of block_tokens
  LatentFormat format = LatentFormat::kBf16;  // kBf16 or kFp8
  int layers() const { return static_cast<int>(kv_heads.size()); }
  bool fp8() const { return format == LatentFormat::kFp8; }
};

// One layer's caches as the kernels see them. bf16: k_cache / v_cache are
// bf16 rows [slots, kv_heads * dim]; fp8: e4m3 code rows [slots, kv_heads *
// dim] (read as bytes) with the scales fp32 [slots, kv_heads].
struct MimoKvCache {
  uint16_t* k_cache = nullptr;      // bf16 [slots, kv_heads * 192] | e4m3 bytes at the same address
  uint16_t* v_cache = nullptr;      // bf16 [slots, kv_heads * 128] | e4m3 bytes
  float* k_scale = nullptr;         // fp8: [slots, kv_heads]
  float* v_scale = nullptr;         // fp8: [slots, kv_heads]
  LatentFormat format = LatentFormat::kBf16;
  int kv_heads = 0;
  const int32_t* block_tables = nullptr;  // int32 [max_requests, blocks_per_request]
  int block_tokens = 0;
  int blocks_per_request = 0;
  bool fp8() const { return format == LatentFormat::kFp8; }
};

class MimoKvPool {
 public:
  MimoKvPool() = default;
  ~MimoKvPool();
  MimoKvPool(const MimoKvPool&) = delete;
  MimoKvPool& operator=(const MimoKvPool&) = delete;

  void init(const MimoKvPoolShape& shape);
  bool initialized() const { return initialized_; }
  static size_t cache_bytes(const MimoKvPoolShape& shape);

  const MimoKvPoolShape& shape() const { return shape_; }
  int64_t total_blocks() const { return table_.total_blocks(); }
  int64_t token_slots() const { return shape_.token_slots; }
  int64_t blocks_in_use() const { return table_.blocks_in_use(); }
  int64_t free_blocks() const { return table_.free_blocks(); }
  int64_t block_count_for_tokens(int64_t tokens) const { return table_.block_count_for_tokens(tokens); }
  const PagedBlockTable& blocks() const { return table_; }

  MimoKvCache view(int layer) const;

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
  // plane a block spans (each layer's K rows, V rows and, under fp8, their
  // scales).
  uint64_t block_identity(int32_t block) const { return table_.block_identity(block); }
  std::vector<CachePlane> planes() const;

 private:
  MimoKvPoolShape shape_;
  bool initialized_ = false;
  PagedBlockTable table_;
  uint16_t* k_base_ = nullptr;   // the layers' K planes, back to back
  uint16_t* v_base_ = nullptr;   // the layers' V planes, back to back
  float* ks_base_ = nullptr;     // fp8: the layers' K scale planes
  float* vs_base_ = nullptr;     // fp8: the layers' V scale planes
  std::vector<size_t> k_off_;    // [layers + 1] BYTE offsets of the K planes
  std::vector<size_t> v_off_;    // [layers + 1]
  std::vector<size_t> s_off_;    // [layers + 1] element offsets of the scale planes (fp8)
  static void plane_offsets(const MimoKvPoolShape& s, std::vector<size_t>& k_off, std::vector<size_t>& v_off,
                            std::vector<size_t>& s_off);
  size_t elem_bytes() const { return shape_.fp8() ? 1 : 2; }
};

}  // namespace dgpp
