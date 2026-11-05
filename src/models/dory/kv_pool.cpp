#include "models/dory/kv_pool.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"

namespace dgpp {

namespace {
void validate(const DoryKvPoolShape& s) {
  if (s.repeats.empty()) throw std::invalid_argument("DoryKvPool: no layers");
  if (s.kv_heads <= 0 || s.dim <= 0 || s.block_tokens <= 0 || s.max_requests <= 0 || s.token_slots <= 0)
    throw std::invalid_argument("DoryKvPool: every shape field must be positive");
  for (int r : s.repeats)
    if (r < 0) throw std::invalid_argument("DoryKvPool: every layer's repeat count must be >= 0");
  if (s.planes() <= 0) throw std::invalid_argument("DoryKvPool: the pool must hold at least one plane");
  if (s.token_slots % s.block_tokens != 0)
    throw std::invalid_argument("DoryKvPool: token_slots must be a multiple of block_tokens");
}
}  // namespace

DoryKvPool::~DoryKvPool() {
  cudaFree(k_base_);
  cudaFree(v_base_);
}

size_t DoryKvPool::cache_bytes(const DoryKvPoolShape& s) {
  validate(s);
  const size_t planes = static_cast<size_t>(s.planes());
  const size_t rows = static_cast<size_t>(s.token_slots) * static_cast<size_t>(s.kv_heads) *
                      static_cast<size_t>(s.dim);
  return 2 * planes * rows * sizeof(uint16_t) +
         PagedBlockTable::table_bytes(s.max_requests, s.token_slots / s.block_tokens);
}

void DoryKvPool::init(const DoryKvPoolShape& shape) {
  if (initialized_) throw std::logic_error("DoryKvPool: init twice");
  validate(shape);
  shape_ = shape;
  k_off_.assign(static_cast<size_t>(shape_.layers()) + 1, 0);
  for (int l = 0; l < shape_.layers(); ++l)
    k_off_[static_cast<size_t>(l) + 1] = k_off_[static_cast<size_t>(l)] +
                                         static_cast<size_t>(shape_.repeats[static_cast<size_t>(l)]);
  const size_t bytes = static_cast<size_t>(shape_.planes()) * plane_kv_elems() * sizeof(uint16_t);
  DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&k_base_), std::max<size_t>(bytes, 1)));
  DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&v_base_), std::max<size_t>(bytes, 1)));
  table_.init(shape_.max_requests, shape_.block_tokens, shape_.token_slots / shape_.block_tokens);
  initialized_ = true;
  reset_all(nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
}

int DoryKvPool::plane_of(int layer, int loop) const {
  if (layer < 0 || layer >= shape_.layers())
    throw std::out_of_range("DoryKvPool: layer " + std::to_string(layer));
  const int r = shape_.repeats[static_cast<size_t>(layer)];
  if (r == 0) throw std::out_of_range("DoryKvPool: layer " + std::to_string(layer) + " has no K/V plane");
  if (loop < 0) loop = 0;
  if (loop >= r) loop = r - 1;
  return static_cast<int>(k_off_[static_cast<size_t>(layer)]) + loop;
}

DoryKvCache DoryKvPool::view_plane(int plane) const {
  if (!initialized_) throw std::logic_error("DoryKvPool: view before init");
  if (plane < 0 || plane >= shape_.planes())
    throw std::out_of_range("DoryKvPool: plane " + std::to_string(plane));
  const size_t off = static_cast<size_t>(plane) * plane_kv_elems();
  DoryKvCache c;
  c.k_cache = k_base_ + off;
  c.v_cache = v_base_ + off;
  c.block_tables = table_.device_tables();
  c.block_tokens = shape_.block_tokens;
  c.blocks_per_request = static_cast<int>(table_.total_blocks());
  c.kv_heads = shape_.kv_heads;
  c.dim = shape_.dim;
  return c;
}

DoryKvCache DoryKvPool::view(int layer, int loop) const {
  return view_plane(plane_of(layer, loop));
}

void DoryKvPool::reset_all(cudaStream_t stream) {
  if (!initialized_) throw std::logic_error("DoryKvPool: reset_all before init");
  const size_t bytes = static_cast<size_t>(shape_.planes()) * plane_kv_elems() * sizeof(uint16_t);
  DGPP_CUDA_OK(cudaMemsetAsync(k_base_, 0, bytes, stream));
  DGPP_CUDA_OK(cudaMemsetAsync(v_base_, 0, bytes, stream));
  table_.reset_all(stream);
}

std::vector<CachePlane> DoryKvPool::planes() const {
  std::vector<CachePlane> out;
  if (!initialized_) return out;
  const size_t blk = static_cast<size_t>(shape_.block_tokens) * kv_row_elems() * sizeof(uint16_t);
  for (int p = 0; p < shape_.planes(); ++p) {
    const DoryKvCache c = view_plane(p);
    out.push_back({reinterpret_cast<uint8_t*>(c.k_cache), blk});
    out.push_back({reinterpret_cast<uint8_t*>(c.v_cache), blk});
  }
  return out;
}

void DoryKvPool::copy_block_contents(int32_t src, int32_t dst, cudaStream_t stream) {
  table_.check_block(src);
  table_.check_block(dst);
  if (src == dst) return;
  const size_t blk = static_cast<size_t>(shape_.block_tokens) * kv_row_elems() * sizeof(uint16_t);
  for (int p = 0; p < shape_.planes(); ++p) {
    const DoryKvCache c = view_plane(p);
    uint8_t* k = reinterpret_cast<uint8_t*>(c.k_cache);
    uint8_t* v = reinterpret_cast<uint8_t*>(c.v_cache);
    DGPP_CUDA_OK(cudaMemcpyAsync(k + static_cast<size_t>(dst) * blk, k + static_cast<size_t>(src) * blk, blk,
                                 cudaMemcpyDeviceToDevice, stream));
    DGPP_CUDA_OK(cudaMemcpyAsync(v + static_cast<size_t>(dst) * blk, v + static_cast<size_t>(src) * blk, blk,
                                 cudaMemcpyDeviceToDevice, stream));
  }
}

}  // namespace dgpp
