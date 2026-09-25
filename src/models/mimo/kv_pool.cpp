#include "models/mimo/kv_pool.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"

namespace dgpp {

namespace {
void validate(const MimoKvPoolShape& s) {
  if (s.layers() <= 0 || s.k_dim <= 0 || s.v_dim <= 0 || s.block_tokens <= 0 || s.max_requests <= 0 ||
      s.token_slots <= 0)
    throw std::invalid_argument("MimoKvPool: every shape field must be positive");
  for (int h : s.kv_heads)
    if (h <= 0) throw std::invalid_argument("MimoKvPool: every layer's kv head count must be positive");
  if (s.token_slots % s.block_tokens != 0)
    throw std::invalid_argument("MimoKvPool: token_slots must be a multiple of block_tokens");
  if (s.format != LatentFormat::kBf16 && s.format != LatentFormat::kFp8)
    throw std::invalid_argument("MimoKvPool: the K/V format must be bf16 or fp8");
}
}  // namespace

MimoKvPool::~MimoKvPool() {
  cudaFree(k_base_);
  cudaFree(v_base_);
  cudaFree(ks_base_);
  cudaFree(vs_base_);
}

// The planes' offsets: bytes for the code / bf16 planes (one or two bytes
// per element), elements for the fp8 scale planes (one fp32 per token and
// kv head; empty under bf16).
void MimoKvPool::plane_offsets(const MimoKvPoolShape& s, std::vector<size_t>& k_off, std::vector<size_t>& v_off,
                               std::vector<size_t>& s_off) {
  const size_t eb = s.fp8() ? 1 : 2;
  k_off.assign(static_cast<size_t>(s.layers()) + 1, 0);
  v_off.assign(static_cast<size_t>(s.layers()) + 1, 0);
  s_off.assign(static_cast<size_t>(s.layers()) + 1, 0);
  for (int l = 0; l < s.layers(); ++l) {
    const size_t heads = static_cast<size_t>(s.kv_heads[static_cast<size_t>(l)]);
    const size_t i = static_cast<size_t>(l);
    k_off[i + 1] = k_off[i] + static_cast<size_t>(s.token_slots) * heads * s.k_dim * eb;
    v_off[i + 1] = v_off[i] + static_cast<size_t>(s.token_slots) * heads * s.v_dim * eb;
    s_off[i + 1] = s_off[i] + (s.fp8() ? static_cast<size_t>(s.token_slots) * heads : 0);
  }
}

size_t MimoKvPool::cache_bytes(const MimoKvPoolShape& s) {
  validate(s);
  std::vector<size_t> k_off, v_off, s_off;
  plane_offsets(s, k_off, v_off, s_off);
  return k_off.back() + v_off.back() + 2 * s_off.back() * sizeof(float) +
         PagedBlockTable::table_bytes(s.max_requests, s.token_slots / s.block_tokens);
}

void MimoKvPool::init(const MimoKvPoolShape& shape) {
  if (initialized_) throw std::logic_error("MimoKvPool: init twice");
  validate(shape);
  shape_ = shape;
  plane_offsets(shape_, k_off_, v_off_, s_off_);
  auto alloc = [](void** p, size_t bytes) { DGPP_CUDA_OK(cudaMalloc(p, std::max<size_t>(bytes, 1))); };
  alloc(reinterpret_cast<void**>(&k_base_), k_off_.back());
  alloc(reinterpret_cast<void**>(&v_base_), v_off_.back());
  if (shape_.fp8()) {
    alloc(reinterpret_cast<void**>(&ks_base_), s_off_.back() * sizeof(float));
    alloc(reinterpret_cast<void**>(&vs_base_), s_off_.back() * sizeof(float));
  }
  table_.init(shape_.max_requests, shape_.block_tokens, shape_.token_slots / shape_.block_tokens);
  initialized_ = true;
  reset_all(nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
}

MimoKvCache MimoKvPool::view(int layer) const {
  if (!initialized_) throw std::logic_error("MimoKvPool: view before init");
  if (layer < 0 || layer >= shape_.layers())
    throw std::out_of_range("MimoKvPool: layer " + std::to_string(layer));
  MimoKvCache c;
  const size_t l = static_cast<size_t>(layer);
  c.k_cache = reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(k_base_) + k_off_[l]);
  c.v_cache = reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(v_base_) + v_off_[l]);
  c.k_scale = shape_.fp8() ? ks_base_ + s_off_[l] : nullptr;
  c.v_scale = shape_.fp8() ? vs_base_ + s_off_[l] : nullptr;
  c.format = shape_.format;
  c.kv_heads = shape_.kv_heads[l];
  c.block_tables = table_.device_tables();
  c.block_tokens = shape_.block_tokens;
  c.blocks_per_request = static_cast<int>(table_.total_blocks());
  return c;
}

void MimoKvPool::reset_all(cudaStream_t stream) {
  if (!initialized_) throw std::logic_error("MimoKvPool: reset_all before init");
  DGPP_CUDA_OK(cudaMemsetAsync(k_base_, 0, k_off_.back(), stream));
  DGPP_CUDA_OK(cudaMemsetAsync(v_base_, 0, v_off_.back(), stream));
  if (shape_.fp8()) {
    DGPP_CUDA_OK(cudaMemsetAsync(ks_base_, 0, s_off_.back() * sizeof(float), stream));
    DGPP_CUDA_OK(cudaMemsetAsync(vs_base_, 0, s_off_.back() * sizeof(float), stream));
  }
  table_.reset_all(stream);
}

std::vector<CachePlane> MimoKvPool::planes() const {
  std::vector<CachePlane> out;
  if (!initialized_) return out;
  const size_t eb = elem_bytes();
  for (int l = 0; l < shape_.layers(); ++l) {
    const MimoKvCache c = view(l);
    const size_t kblk = static_cast<size_t>(shape_.block_tokens) * static_cast<size_t>(c.kv_heads) * shape_.k_dim * eb;
    const size_t vblk = static_cast<size_t>(shape_.block_tokens) * static_cast<size_t>(c.kv_heads) * shape_.v_dim * eb;
    out.push_back({reinterpret_cast<uint8_t*>(c.k_cache), kblk});
    out.push_back({reinterpret_cast<uint8_t*>(c.v_cache), vblk});
    if (shape_.fp8()) {
      const size_t sblk = static_cast<size_t>(shape_.block_tokens) * static_cast<size_t>(c.kv_heads) * sizeof(float);
      out.push_back({reinterpret_cast<uint8_t*>(c.k_scale), sblk});
      out.push_back({reinterpret_cast<uint8_t*>(c.v_scale), sblk});
    }
  }
  return out;
}

void MimoKvPool::copy_block_contents(int32_t src, int32_t dst, cudaStream_t stream) {
  table_.check_block(src);
  table_.check_block(dst);
  if (src == dst) return;
  const size_t eb = elem_bytes();
  for (int l = 0; l < shape_.layers(); ++l) {
    const MimoKvCache c = view(l);
    const size_t kblk = static_cast<size_t>(shape_.block_tokens) * static_cast<size_t>(c.kv_heads) * shape_.k_dim * eb;
    const size_t vblk = static_cast<size_t>(shape_.block_tokens) * static_cast<size_t>(c.kv_heads) * shape_.v_dim * eb;
    uint8_t* k = reinterpret_cast<uint8_t*>(c.k_cache);
    uint8_t* v = reinterpret_cast<uint8_t*>(c.v_cache);
    DGPP_CUDA_OK(cudaMemcpyAsync(k + static_cast<size_t>(dst) * kblk, k + static_cast<size_t>(src) * kblk, kblk,
                                 cudaMemcpyDeviceToDevice, stream));
    DGPP_CUDA_OK(cudaMemcpyAsync(v + static_cast<size_t>(dst) * vblk, v + static_cast<size_t>(src) * vblk, vblk,
                                 cudaMemcpyDeviceToDevice, stream));
    if (shape_.fp8()) {
      const size_t sblk = static_cast<size_t>(shape_.block_tokens) * static_cast<size_t>(c.kv_heads);
      DGPP_CUDA_OK(cudaMemcpyAsync(c.k_scale + static_cast<size_t>(dst) * sblk, c.k_scale + static_cast<size_t>(src) * sblk,
                                   sblk * sizeof(float), cudaMemcpyDeviceToDevice, stream));
      DGPP_CUDA_OK(cudaMemcpyAsync(c.v_scale + static_cast<size_t>(dst) * sblk, c.v_scale + static_cast<size_t>(src) * sblk,
                                   sblk * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    }
  }
}

}  // namespace dgpp
