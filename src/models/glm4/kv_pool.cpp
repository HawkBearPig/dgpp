#include "models/glm4/kv_pool.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"

namespace dgpp {

namespace {
void validate(const Glm4KvPoolShape& s) {
  if (s.layers < 0 || s.kv_heads <= 0 || s.dim <= 0 || s.block_tokens <= 0 || s.max_requests <= 0 ||
      s.token_slots <= 0)
    throw std::invalid_argument("Glm4KvPool: every shape field must be positive");
  if (s.token_slots % s.block_tokens != 0)
    throw std::invalid_argument("Glm4KvPool: token_slots must be a multiple of block_tokens");
}
}  // namespace

Glm4KvPool::~Glm4KvPool() {
  cudaFree(k_base_);
  cudaFree(v_base_);
}

size_t Glm4KvPool::cache_bytes(const Glm4KvPoolShape& s) {
  validate(s);
  const size_t L = static_cast<size_t>(s.layers);
  const size_t kv = static_cast<size_t>(s.token_slots) * s.kv_heads * s.dim * 2;
  return L * 2 * kv + PagedBlockTable::table_bytes(s.max_requests, s.token_slots / s.block_tokens);
}

void Glm4KvPool::init(const Glm4KvPoolShape& shape) {
  if (initialized_) throw std::logic_error("Glm4KvPool: init twice");
  validate(shape);
  shape_ = shape;
  const size_t L = static_cast<size_t>(shape_.layers);
  auto alloc = [](void** p, size_t bytes) { DGPP_CUDA_OK(cudaMalloc(p, std::max<size_t>(bytes, 1))); };
  alloc(reinterpret_cast<void**>(&k_base_), L * layer_kv_elems() * 2);
  alloc(reinterpret_cast<void**>(&v_base_), L * layer_kv_elems() * 2);
  table_.init(shape_.max_requests, shape_.block_tokens, shape_.token_slots / shape_.block_tokens);
  initialized_ = true;
  reset_all(nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
}

Glm4KvCache Glm4KvPool::view(int layer) const {
  if (!initialized_) throw std::logic_error("Glm4KvPool: view before init");
  if (layer < 0 || layer >= shape_.layers)
    throw std::out_of_range("Glm4KvPool: layer " + std::to_string(layer));
  Glm4KvCache c;
  const size_t l = static_cast<size_t>(layer);
  c.k_cache = k_base_ + l * layer_kv_elems();
  c.v_cache = v_base_ + l * layer_kv_elems();
  c.block_tables = table_.device_tables();
  c.block_tokens = shape_.block_tokens;
  c.blocks_per_request = static_cast<int>(table_.total_blocks());
  return c;
}

void Glm4KvPool::reset_all(cudaStream_t stream) {
  if (!initialized_) throw std::logic_error("Glm4KvPool: reset_all before init");
  const size_t L = static_cast<size_t>(shape_.layers);
  DGPP_CUDA_OK(cudaMemsetAsync(k_base_, 0, L * layer_kv_elems() * 2, stream));
  DGPP_CUDA_OK(cudaMemsetAsync(v_base_, 0, L * layer_kv_elems() * 2, stream));
  table_.reset_all(stream);
}

std::vector<CachePlane> Glm4KvPool::planes() const {
  std::vector<CachePlane> out;
  if (!initialized_) return out;
  const size_t blk = static_cast<size_t>(shape_.block_tokens) * kv_row_elems() * 2;
  for (int l = 0; l < shape_.layers; ++l) {
    const Glm4KvCache c = view(l);
    out.push_back({reinterpret_cast<uint8_t*>(c.k_cache), blk});
    out.push_back({reinterpret_cast<uint8_t*>(c.v_cache), blk});
  }
  return out;
}

void Glm4KvPool::copy_block_contents(int32_t src, int32_t dst, cudaStream_t stream) {
  table_.check_block(src);
  table_.check_block(dst);
  if (src == dst) return;
  const size_t blk = static_cast<size_t>(shape_.block_tokens) * kv_row_elems();
  for (int l = 0; l < shape_.layers; ++l) {
    const Glm4KvCache c = view(l);
    DGPP_CUDA_OK(cudaMemcpyAsync(c.k_cache + static_cast<size_t>(dst) * blk, c.k_cache + static_cast<size_t>(src) * blk,
                                 blk * 2, cudaMemcpyDeviceToDevice, stream));
    DGPP_CUDA_OK(cudaMemcpyAsync(c.v_cache + static_cast<size_t>(dst) * blk, c.v_cache + static_cast<size_t>(src) * blk,
                                 blk * 2, cudaMemcpyDeviceToDevice, stream));
  }
}

}  // namespace dgpp
