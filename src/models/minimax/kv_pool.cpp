#include "models/minimax/kv_pool.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"

namespace dgpp {

namespace {
void validate(const MinimaxKvPoolShape& s) {
  if (s.layers <= 0 || s.kv_heads <= 0 || s.dim != 128 || s.block_tokens != 128 ||
      s.max_requests <= 0 || s.token_slots <= 0)
    throw std::invalid_argument("MinimaxKvPool: every shape field must be positive");
  if (s.token_slots % s.block_tokens != 0)
    throw std::invalid_argument("MinimaxKvPool: token_slots must be a multiple of block_tokens");
}
}  // namespace

MinimaxKvPool::~MinimaxKvPool() {
  cudaFree(k_base_);
  cudaFree(v_base_);
  cudaFree(index_base_);
}

size_t MinimaxKvPool::cache_bytes(const MinimaxKvPoolShape& s) {
  validate(s);
  const size_t L = static_cast<size_t>(s.layers);
  const size_t kv = static_cast<size_t>(s.token_slots) * s.kv_heads * s.dim * 2;
  return L * (2 * kv + static_cast<size_t>(s.token_slots) * 128 * 2) +
         PagedBlockTable::table_bytes(s.max_requests, s.token_slots / s.block_tokens);
}

void MinimaxKvPool::init(const MinimaxKvPoolShape& shape) {
  if (initialized_) throw std::logic_error("MinimaxKvPool: init twice");
  validate(shape);
  shape_ = shape;
  const size_t L = static_cast<size_t>(shape_.layers);
  auto alloc = [](void** p, size_t bytes) {
    DGPP_CUDA_OK(cudaMalloc(p, std::max<size_t>(bytes, 1)));
  };
  alloc(reinterpret_cast<void**>(&k_base_), L * layer_kv_elems() * 2);
  alloc(reinterpret_cast<void**>(&v_base_), L * layer_kv_elems() * 2);
  alloc(reinterpret_cast<void**>(&index_base_),
        L * static_cast<size_t>(shape_.token_slots) * 128 * 2);
  table_.init(shape_.max_requests, shape_.block_tokens, shape_.token_slots / shape_.block_tokens);
  initialized_ = true;
  reset_all(nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
}

MinimaxKvCache MinimaxKvPool::view(int layer) const {
  if (!initialized_) throw std::logic_error("MinimaxKvPool: view before init");
  if (layer < 0 || layer >= shape_.layers)
    throw std::out_of_range("MinimaxKvPool: layer " + std::to_string(layer));
  MinimaxKvCache c;
  const size_t l = static_cast<size_t>(layer);
  c.k_cache = k_base_ + l * layer_kv_elems();
  c.v_cache = v_base_ + l * layer_kv_elems();
  c.index_cache = index_base_ + l * static_cast<size_t>(shape_.token_slots) * 128;
  c.block_tables = table_.device_tables();
  c.block_tokens = shape_.block_tokens;
  c.blocks_per_request = static_cast<int>(table_.total_blocks());
  return c;
}

void MinimaxKvPool::reset_all(cudaStream_t stream) {
  if (!initialized_) throw std::logic_error("MinimaxKvPool: reset_all before init");
  const size_t L = static_cast<size_t>(shape_.layers);
  DGPP_CUDA_OK(cudaMemsetAsync(k_base_, 0, L * layer_kv_elems() * 2, stream));
  DGPP_CUDA_OK(cudaMemsetAsync(v_base_, 0, L * layer_kv_elems() * 2, stream));
  DGPP_CUDA_OK(cudaMemsetAsync(index_base_, 0,
                               L * static_cast<size_t>(shape_.token_slots) * 128 * 2, stream));
  table_.reset_all(stream);
}

std::vector<CachePlane> MinimaxKvPool::planes() const {
  std::vector<CachePlane> out;
  if (!initialized_) return out;
  const size_t blk = static_cast<size_t>(shape_.block_tokens) * kv_row_elems() * 2;
  for (int l = 0; l < shape_.layers; ++l) {
    const MinimaxKvCache c = view(l);
    out.push_back({reinterpret_cast<uint8_t*>(c.k_cache), blk});
    out.push_back({reinterpret_cast<uint8_t*>(c.v_cache), blk});
    out.push_back({reinterpret_cast<uint8_t*>(c.index_cache),
                   static_cast<size_t>(shape_.block_tokens) * 128 * 2});
  }
  return out;
}

void MinimaxKvPool::copy_block_contents(int32_t src, int32_t dst, cudaStream_t stream) {
  table_.check_block(src);
  table_.check_block(dst);
  if (src == dst) return;
  for (const auto& plane : planes()) {
    DGPP_CUDA_OK(cudaMemcpyAsync(plane.base + static_cast<size_t>(dst) * plane.block_bytes,
                                 plane.base + static_cast<size_t>(src) * plane.block_bytes,
                                 plane.block_bytes, cudaMemcpyDeviceToDevice, stream));
  }
}

}  // namespace dgpp
