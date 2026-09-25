#pragma once
// The paged cache's block management (2026-09-09, extracted from the Qwen
// pool for the second paged family, GLM-4.7): one block table int32
// [max_requests, total_blocks] shared by every cache plane of a model —
// block b of request r holds the request's tokens [b*block_tokens,
// (b+1)*block_tokens) in every plane through one physical block id
// (blocks_per_request == total_blocks: one request may take the whole
// pool). Host-side: a LIFO free list, a mirrored table, and REFCOUNTS (a
// request row holds one reference per block, a prefix-cache entry pins its
// blocks with one more; a block returns to the free list at zero).
// Acquired blocks are not scrubbed on release: a new owner writes every
// row it reads (the prefill appends before any attention touches them), so
// release never lands on the memory hot path. The planes themselves (K/V
// rows, index keys, rings) are the family pool's; this owns the table.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"

namespace dgpp {

class PagedBlockTable {
 public:
  PagedBlockTable() = default;
  ~PagedBlockTable() { cudaFree(tables_); }
  PagedBlockTable(const PagedBlockTable&) = delete;
  PagedBlockTable& operator=(const PagedBlockTable&) = delete;

  // The device table's bytes for a shape (the memory plan).
  static size_t table_bytes(int max_requests, int64_t total_blocks) {
    return static_cast<size_t>(max_requests) * static_cast<size_t>(total_blocks) * 4;
  }

  void init(int max_requests, int block_tokens, int64_t total_blocks) {
    if (initialized_) throw std::logic_error("PagedBlockTable: init twice");
    if (max_requests <= 0 || block_tokens <= 0 || total_blocks <= 0)
      throw std::invalid_argument("PagedBlockTable: every shape field must be positive");
    max_requests_ = max_requests;
    block_tokens_ = block_tokens;
    total_blocks_ = total_blocks;
    const size_t n = static_cast<size_t>(max_requests_) * static_cast<size_t>(total_blocks_);
    DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&tables_), std::max<size_t>(n * 4, 16)));
    tables_host_.assign(n, 0);
    held_.assign(static_cast<size_t>(max_requests_), 0);
    refcount_.assign(static_cast<size_t>(total_blocks_), 0);
    generation_.assign(static_cast<size_t>(total_blocks_), 0);
    initialized_ = true;
    reset_all(nullptr);
  }
  bool initialized() const { return initialized_; }

  int max_requests() const { return max_requests_; }
  int block_tokens() const { return block_tokens_; }
  int64_t total_blocks() const { return total_blocks_; }
  int64_t token_slots() const { return total_blocks_ * block_tokens_; }
  int64_t blocks_in_use() const { return total_blocks_ - static_cast<int64_t>(free_.size()); }
  int64_t free_blocks() const { return static_cast<int64_t>(free_.size()); }
  int64_t block_count_for_tokens(int64_t tokens) const {
    return (tokens + block_tokens_ - 1) / block_tokens_;
  }
  // The device table [max_requests, total_blocks] the kernels index.
  const int32_t* device_tables() const { return tables_; }
  int32_t* device_tables() { return tables_; }

  // Transactionally grows req's table to cover `tokens` tokens, acquiring
  // physical blocks and uploading the new table slice on `stream`. False
  // without side effects when the pool cannot satisfy the request.
  bool ensure_request_blocks(int req, int64_t tokens, cudaStream_t stream) {
    check_req(req, "ensure_request_blocks");
    if (tokens < 0) throw std::invalid_argument("PagedBlockTable: negative token count");
    const int64_t needed = block_count_for_tokens(tokens);
    if (needed > total_blocks_) return false;
    const int64_t have = held_[static_cast<size_t>(req)];
    if (needed <= have) return true;
    const int64_t extra = needed - have;
    if (static_cast<int64_t>(free_.size()) < extra) return false;  // transactional
    int32_t* row = tables_host_.data() + static_cast<size_t>(req) * total_blocks_;
    for (int64_t i = 0; i < extra; ++i) {
      row[have + i] = free_.back();
      free_.pop_back();
      refcount_[static_cast<size_t>(row[have + i])] = 1;
      ++generation_[static_cast<size_t>(row[have + i])];
    }
    held_[static_cast<size_t>(req)] = static_cast<int32_t>(needed);
    DGPP_CUDA_OK(cudaMemcpyAsync(tables_ + static_cast<size_t>(req) * total_blocks_ + have, row + have,
                                 static_cast<size_t>(extra) * 4, cudaMemcpyHostToDevice, stream));
    return true;
  }

  // Drops req's references and zeroes its table row (uploaded on `stream`).
  void release_request_blocks(int req, cudaStream_t stream) {
    check_req(req, "release_request_blocks");
    const int64_t held = held_[static_cast<size_t>(req)];
    if (held == 0) return;
    int32_t* row = tables_host_.data() + static_cast<size_t>(req) * total_blocks_;
    for (int64_t i = held - 1; i >= 0; --i) {
      const int32_t b = row[i];
      if (--refcount_[static_cast<size_t>(b)] == 0) free_.push_back(b);
    }
    std::fill(row, row + held, 0);
    held_[static_cast<size_t>(req)] = 0;
    DGPP_CUDA_OK(cudaMemcpyAsync(tables_ + static_cast<size_t>(req) * total_blocks_, row,
                                 static_cast<size_t>(held) * 4, cudaMemcpyHostToDevice, stream));
  }

  int64_t request_blocks(int req) const {
    check_req(req, "request_blocks");
    return held_[static_cast<size_t>(req)];
  }
  const int32_t* request_table_row(int req) const {
    check_req(req, "request_table_row");
    return tables_host_.data() + static_cast<size_t>(req) * total_blocks_;
  }

  // Cold start: every table row zero, all blocks free (the planes are the
  // family pool's to clear).
  void reset_all(cudaStream_t stream) {
    if (!initialized_) throw std::logic_error("PagedBlockTable: reset_all before init");
    std::fill(tables_host_.begin(), tables_host_.end(), 0);
    DGPP_CUDA_OK(cudaMemsetAsync(tables_, 0, tables_host_.size() * 4, stream));
    std::fill(held_.begin(), held_.end(), 0);
    std::fill(refcount_.begin(), refcount_.end(), 0);
    free_.clear();
    for (int64_t b = total_blocks_ - 1; b >= 0; --b) free_.push_back(static_cast<int32_t>(b));
  }

  // ---- sharing (the prefix cache) --------------------------------------
  // A FRESH row takes `n` live physical blocks by reference as its logical
  // blocks [0, n) — the shared immutable prefix. False when n exceeds the
  // pool.
  bool share_blocks_into(int req, const int32_t* blocks, int64_t n, cudaStream_t stream) {
    check_req(req, "share_blocks_into");
    if (held_[static_cast<size_t>(req)] != 0)
      throw std::logic_error("PagedBlockTable: share_blocks_into needs a fresh row");
    if (n < 0 || n > total_blocks_) return false;
    if (n == 0) return true;
    int32_t* row = tables_host_.data() + static_cast<size_t>(req) * total_blocks_;
    for (int64_t i = 0; i < n; ++i) {
      const int32_t b = blocks[i];
      if (b < 0 || b >= total_blocks_ || refcount_[static_cast<size_t>(b)] <= 0)
        throw std::logic_error("PagedBlockTable: sharing a block that is not live");
      row[i] = b;
      ++refcount_[static_cast<size_t>(b)];
    }
    held_[static_cast<size_t>(req)] = static_cast<int32_t>(n);
    DGPP_CUDA_OK(cudaMemcpyAsync(tables_ + static_cast<size_t>(req) * total_blocks_, row,
                                 static_cast<size_t>(n) * 4, cudaMemcpyHostToDevice, stream));
    return true;
  }
  void pin_blocks(const int32_t* blocks, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
      const int32_t b = blocks[i];
      if (b < 0 || b >= total_blocks_ || refcount_[static_cast<size_t>(b)] <= 0)
        throw std::logic_error("PagedBlockTable: pinning a block that is not live");
      ++refcount_[static_cast<size_t>(b)];
    }
  }
  void unpin_blocks(const int32_t* blocks, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
      const int32_t b = blocks[i];
      if (b < 0 || b >= total_blocks_ || refcount_[static_cast<size_t>(b)] <= 0)
        throw std::logic_error("PagedBlockTable: unpinning a block that is not live");
      if (--refcount_[static_cast<size_t>(b)] == 0) free_.push_back(b);
    }
  }
  // A fresh block owned by the caller (refcount 1, no row); -1 when empty.
  int32_t acquire_pinned_block() {
    if (free_.empty()) return -1;
    const int32_t b = free_.back();
    free_.pop_back();
    refcount_[static_cast<size_t>(b)] = 1;
    ++generation_[static_cast<size_t>(b)];
    return b;
  }
  int32_t block_refcount(int32_t block) const { return refcount_[static_cast<size_t>(block)]; }
  // The block's identity for the NVMe cold tier (issue #26): its id and
  // the generation of its current contents — bumped at every acquisition,
  // so a freed and reused block never passes for the state it held. Two
  // prefix entries holding the same identity hold the same bytes (the
  // shared immutable prefix), which is what lets the tier store them once.
  uint64_t block_identity(int32_t block) const {
    check_block(block);
    return (generation_[static_cast<size_t>(block)] << 32) | static_cast<uint32_t>(block);
  }
  void check_block(int32_t block) const {
    if (block < 0 || block >= total_blocks_)
      throw std::out_of_range("PagedBlockTable: block index out of range");
  }

 private:
  void check_req(int req, const char* what) const {
    if (!initialized_) throw std::logic_error(std::string("PagedBlockTable: ") + what + " before init");
    if (req < 0 || req >= max_requests_)
      throw std::out_of_range(std::string("PagedBlockTable: ") + what + " request " + std::to_string(req));
  }

  bool initialized_ = false;
  int max_requests_ = 0;
  int block_tokens_ = 0;
  int64_t total_blocks_ = 0;
  int32_t* tables_ = nullptr;  // device [max_requests][total_blocks]
  std::vector<int32_t> tables_host_;
  std::vector<int32_t> held_;  // blocks per request row
  std::vector<int32_t> free_;  // LIFO
  std::vector<int32_t> refcount_;
  std::vector<uint64_t> generation_;  // per block: acquisitions so far
};

}  // namespace dgpp
