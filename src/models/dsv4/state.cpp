#include "models/dsv4/state.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"
#include "kernels/csa2.hpp"

namespace dgpp {
namespace {
size_t padded(size_t b) { return (b + 255) / 256 * 256; }
void check_shape(const Dsv4PoolShape& s) {
  if (s.layers <= 0 || s.max_requests <= 0 || s.token_slots <= 0 || s.block_tokens <= 0 ||
      s.token_slots % s.block_tokens != 0 || s.ring_slots <= 0)
    throw std::invalid_argument("dsv4 state pool: invalid shape");
  for (const int r : s.cache_ratio)
    if ((r != 4 && r != 128) || s.block_tokens % r != 0)
      throw std::invalid_argument("dsv4 state pool: a cache ratio must be 4 or 128 and divide the block");
  if (s.lazy_hidden < 0 || s.lazy_hidden % 8 != 0)
    throw std::invalid_argument("dsv4 state pool: lazy_hidden must be a multiple of 8");
  if (s.token_slots >= (int64_t(1) << 23))
    throw std::invalid_argument("dsv4 state pool: entry id space exceeds 2^21 (the select keys)");
}
template <class T>
T* dev_alloc(size_t n) {
  T* p = nullptr;
  DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&p), std::max<size_t>(n * sizeof(T), 16)));
  return p;
}
}  // namespace

Dsv4StatePool::~Dsv4StatePool() {
  for (uint8_t* p : main_) cudaFree(p);
  for (uint8_t* p : index_k_) cudaFree(p);
  for (float* p : index_scale_) cudaFree(p);
  for (float* p : comp_ring_) cudaFree(p);
  for (uint16_t* p : u_ring_) cudaFree(p);
  for (float* p : index_comp_ring_) cudaFree(p);
  cudaFree(ring_base_);
  cudaFree(ring_table_);
  cudaFree(ring_table_wide_);
}

void Dsv4StatePool::init(const Dsv4PoolShape& shape) {
  if (initialized_) throw std::logic_error("dsv4 state pool: init twice");
  check_shape(shape);
  shape_ = shape;
  table_.init(shape.max_requests, shape.block_tokens, shape.token_slots / shape.block_tokens);
  const size_t n = static_cast<size_t>(caches());
  const size_t R = static_cast<size_t>(shape.max_requests);
  main_.assign(n, nullptr);
  index_k_.assign(n, nullptr);
  index_scale_.assign(n, nullptr);
  comp_ring_.assign(n, nullptr);
  u_ring_.assign(n, nullptr);
  index_comp_ring_.assign(n, nullptr);
  for (size_t o = 0; o < n; ++o) {
    const int ord = static_cast<int>(o);
    const size_t slots = static_cast<size_t>(entry_slots(ord));
    main_[o] = dev_alloc<uint8_t>(slots * kRowBytes);
    if (lazy(ord))
      u_ring_[o] = dev_alloc<uint16_t>(R * u_ring_elems_per_request());
    else
      comp_ring_[o] = dev_alloc<float>(R * main_geom(cache_ratio(ord)).ring_elems());
    if (indexed(ord)) {
      index_k_[o] = dev_alloc<uint8_t>(slots * kDsv4IndexDim);
      index_scale_[o] = dev_alloc<float>(slots);
      index_comp_ring_[o] = dev_alloc<float>(R * index_geom(cache_ratio(ord)).ring_elems());
    }
  }
  ring_base_ = dev_alloc<uint8_t>(static_cast<size_t>(shape.layers) * R * ring_bytes_per_request());
  ring_table_ = dev_alloc<int32_t>(R);
  csa2_ring_table(ring_table_, shape.max_requests, nullptr);
  {
    std::vector<int32_t> wide(R * R);
    for (size_t r = 0; r < R; ++r)
      for (size_t j = 0; j < R; ++j) wide[r * R + j] = static_cast<int32_t>(r);
    ring_table_wide_ = dev_alloc<int32_t>(R * R);
    DGPP_CUDA_OK(cudaMemcpy(ring_table_wide_, wide.data(), wide.size() * 4, cudaMemcpyHostToDevice));
  }
  initialized_ = true;
  reset_all(nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
}

size_t Dsv4StatePool::cache_bytes(const Dsv4PoolShape& s) {
  check_shape(s);
  const int64_t blocks = s.token_slots / s.block_tokens;
  const size_t R = static_cast<size_t>(s.max_requests);
  size_t total = 0;
  for (const int r : s.cache_ratio) {
    const size_t slots = static_cast<size_t>(blocks) * static_cast<size_t>(s.block_tokens / r);
    total += padded(slots * kRowBytes);
    if (r == 128 && s.lazy_hidden > 0)
      total += padded(R * static_cast<size_t>(main_geom(r).slots) * static_cast<size_t>(s.lazy_hidden) * 2);
    else
      total += padded(R * main_geom(r).ring_elems() * sizeof(float));
    if (r == 4) {
      total += padded(slots * kDsv4IndexDim);
      total += padded(slots * sizeof(float));
      total += padded(R * index_geom(r).ring_elems() * sizeof(float));
    }
  }
  total += padded(static_cast<size_t>(s.layers) * R * s.ring_slots * kRowBytes);
  total += padded(R * 4);
  total += padded(PagedBlockTable::table_bytes(s.max_requests, blocks));
  return total;
}

void Dsv4StatePool::check_ord(int ord, const char* what) const {
  if (!initialized_) throw std::logic_error(std::string("dsv4 state pool: ") + what + " before init");
  if (ord < 0 || ord >= caches())
    throw std::out_of_range(std::string("dsv4 state pool: ") + what + " cache ordinal " + std::to_string(ord));
}
uint8_t* Dsv4StatePool::main(int ord) const {
  check_ord(ord, "main");
  return main_[static_cast<size_t>(ord)];
}
uint8_t* Dsv4StatePool::index_k(int ord) const {
  check_ord(ord, "index_k");
  if (!indexed(ord)) throw std::logic_error("dsv4 state pool: index_k of a cache without an indexer");
  return index_k_[static_cast<size_t>(ord)];
}
float* Dsv4StatePool::index_scale(int ord) const {
  check_ord(ord, "index_scale");
  if (!indexed(ord)) throw std::logic_error("dsv4 state pool: index_scale of a cache without an indexer");
  return index_scale_[static_cast<size_t>(ord)];
}
uint16_t* Dsv4StatePool::u_ring(int ord) const {
  check_ord(ord, "u_ring");
  if (!lazy(ord)) throw std::logic_error("dsv4 state pool: u_ring of a cache that is not lazily pooled");
  return u_ring_[static_cast<size_t>(ord)];
}
float* Dsv4StatePool::comp_ring(int ord) const {
  check_ord(ord, "comp_ring");
  if (lazy(ord)) throw std::logic_error("dsv4 state pool: comp_ring of a lazily pooled cache");
  return comp_ring_[static_cast<size_t>(ord)];
}
float* Dsv4StatePool::index_comp_ring(int ord) const {
  check_ord(ord, "index_comp_ring");
  if (!indexed(ord)) throw std::logic_error("dsv4 state pool: index_comp_ring of a cache without an indexer");
  return index_comp_ring_[static_cast<size_t>(ord)];
}
uint8_t* Dsv4StatePool::ring(int layer) const {
  if (!initialized_ || layer < 0 || layer >= shape_.layers)
    throw std::out_of_range("dsv4 state pool: ring layer " + std::to_string(layer));
  return ring_base_ + static_cast<size_t>(layer) * shape_.max_requests * ring_bytes_per_request();
}

void Dsv4StatePool::reset_request(int req, cudaStream_t stream) {
  if (!initialized_) throw std::logic_error("dsv4 state pool: reset_request before init");
  if (req < 0 || req >= shape_.max_requests) throw std::out_of_range("dsv4 state pool: request");
  const size_t r = static_cast<size_t>(req);
  for (int l = 0; l < shape_.layers; ++l)
    DGPP_CUDA_OK(cudaMemsetAsync(ring(l) + r * ring_bytes_per_request(), 0, ring_bytes_per_request(), stream));
  for (int o = 0; o < caches(); ++o) {
    // A lazy ring needs no reset: a group's rows are all rewritten before it pools.
    if (lazy(o)) continue;
    const size_t cb = comp_ring_bytes_per_request(o);
    DGPP_CUDA_OK(cudaMemsetAsync(reinterpret_cast<uint8_t*>(comp_ring(o)) + r * cb, 0, cb, stream));
    if (indexed(o)) {
      const size_t ib = index_comp_ring_bytes_per_request(o);
      DGPP_CUDA_OK(cudaMemsetAsync(reinterpret_cast<uint8_t*>(index_comp_ring(o)) + r * ib, 0, ib, stream));
    }
  }
  table_.release_request_blocks(req, stream);
}

void Dsv4StatePool::reset_all(cudaStream_t stream) {
  if (!initialized_) throw std::logic_error("dsv4 state pool: reset_all before init");
  const size_t R = static_cast<size_t>(shape_.max_requests);
  for (int o = 0; o < caches(); ++o) {
    const size_t i = static_cast<size_t>(o);
    const size_t slots = static_cast<size_t>(entry_slots(o));
    DGPP_CUDA_OK(cudaMemsetAsync(main_[i], 0, slots * kRowBytes, stream));
    if (lazy(o))
      DGPP_CUDA_OK(cudaMemsetAsync(u_ring_[i], 0, R * u_ring_elems_per_request() * 2, stream));
    else
      DGPP_CUDA_OK(cudaMemsetAsync(comp_ring_[i], 0, R * comp_ring_bytes_per_request(o), stream));
    if (indexed(o)) {
      DGPP_CUDA_OK(cudaMemsetAsync(index_k_[i], 0, slots * kDsv4IndexDim, stream));
      DGPP_CUDA_OK(cudaMemsetAsync(index_scale_[i], 0, slots * sizeof(float), stream));
      DGPP_CUDA_OK(cudaMemsetAsync(index_comp_ring_[i], 0, R * index_comp_ring_bytes_per_request(o), stream));
    }
  }
  DGPP_CUDA_OK(cudaMemsetAsync(ring_base_, 0, static_cast<size_t>(shape_.layers) * R * ring_bytes_per_request(), stream));
  table_.reset_all(stream);
}

std::vector<CachePlane> Dsv4StatePool::planes() const {
  std::vector<CachePlane> out;
  if (!initialized_) return out;
  for (int o = 0; o < caches(); ++o) {
    const size_t i = static_cast<size_t>(o);
    const size_t epb = static_cast<size_t>(entries_per_block(o));
    out.push_back({main_[i], epb * kRowBytes});
    if (!indexed(o)) continue;
    out.push_back({index_k_[i], epb * kDsv4IndexDim});
    out.push_back({reinterpret_cast<uint8_t*>(index_scale_[i]), epb * sizeof(float)});
  }
  return out;
}

void Dsv4StatePool::copy_block_contents(int32_t src, int32_t dst, cudaStream_t stream) {
  table_.check_block(src);
  table_.check_block(dst);
  if (src == dst) return;
  for (int o = 0; o < caches(); ++o) {
    const size_t i = static_cast<size_t>(o);
    const size_t epb = static_cast<size_t>(entries_per_block(o));
    const size_t mb = epb * kRowBytes;
    DGPP_CUDA_OK(cudaMemcpyAsync(main_[i] + static_cast<size_t>(dst) * mb, main_[i] + static_cast<size_t>(src) * mb, mb,
                                 cudaMemcpyDeviceToDevice, stream));
    if (!indexed(o)) continue;
    const size_t kb = epb * kDsv4IndexDim;
    DGPP_CUDA_OK(cudaMemcpyAsync(index_k_[i] + static_cast<size_t>(dst) * kb, index_k_[i] + static_cast<size_t>(src) * kb,
                                 kb, cudaMemcpyDeviceToDevice, stream));
    DGPP_CUDA_OK(cudaMemcpyAsync(index_scale_[i] + static_cast<size_t>(dst) * epb,
                                 index_scale_[i] + static_cast<size_t>(src) * epb, epb * sizeof(float),
                                 cudaMemcpyDeviceToDevice, stream));
  }
}

}  // namespace dgpp
