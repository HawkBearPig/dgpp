#include "models/dsa_state.hpp"

#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"

namespace dgpp {

namespace {

const size_t kRegionAlign = 256;

size_t padded(size_t bytes) { return (bytes + kRegionAlign - 1) / kRegionAlign * kRegionAlign; }

}  // namespace

void DsaStatePool::init(Arena& arena, const DsaConfig& cfg, int max_requests,
                        int64_t max_token_slots, const std::vector<int>& index_ordinal) {
  if (initialized_) throw std::logic_error("dsa state pool is already initialized");
  DsaConfig::validate_config(cfg);
  // The layer -> index-cache map: the identity, or the caller's table with
  // every ordinal used once.
  const int index_layers = cfg.index_layers();
  if (index_ordinal.empty()) {
    if (index_layers != cfg.num_dsa_layers)
      throw std::invalid_argument(
          "dsa state pool: a config with fewer index layers than DSA layers needs "
          "the layer -> index ordinal table");
    index_ordinal_.resize(size_t(cfg.num_dsa_layers));
    for (int l = 0; l < cfg.num_dsa_layers; ++l) index_ordinal_[size_t(l)] = l;
  } else {
    if (int(index_ordinal.size()) != cfg.num_dsa_layers)
      throw std::invalid_argument("dsa state pool: index ordinal table size");
    std::vector<int> seen(size_t(index_layers), 0);
    for (int o : index_ordinal) {
      if (o < 0) continue;
      if (o >= index_layers || seen[size_t(o)]++)
        throw std::invalid_argument("dsa state pool: index ordinal table must use each "
                                    "ordinal in [0, index_layers) once");
    }
    for (int o = 0; o < index_layers; ++o)
      if (!seen[size_t(o)])
        throw std::invalid_argument("dsa state pool: index ordinal table leaves an "
                                    "index cache unowned");
    index_ordinal_ = index_ordinal;
  }
  if (max_requests <= 0)
    throw std::invalid_argument("dsa state pool: max_requests must be positive");
  if (max_token_slots <= 0)
    throw std::invalid_argument("dsa state pool: max_token_slots must be positive");
  if (max_token_slots % cfg.block_tokens != 0)
    throw std::invalid_argument(
        "dsa state pool: max_token_slots must be a multiple of block_tokens");

  cfg_ = cfg;
  geo_ = DsaGeometry::from_config(cfg);
  max_requests_ = max_requests;
  max_token_slots_ = max_token_slots;
  total_blocks_ = max_token_slots / cfg.block_tokens;
  max_pool_slots_ = total_blocks_ * geo_.pools_per_block;
  // The selection kernels pack pool ids into 21 composite-key bits; a pool
  // id at or above 2^21 would alias into the logit bits.
  if (max_pool_slots_ >= (int64_t(1) << 21))
    throw std::invalid_argument("dsa state pool: pool id space exceeds 2^21");

  const int layers = cfg.num_dsa_layers;
  const int64_t pools = max_pool_slots_;
  const int ilayers = index_layers;
  // One region per array (layer-major inside each): the caches have no
  // snapshot/copy requirement in M3 (prefix attachment shares by reference,
  // DESIGN §8), so contiguity across regions buys nothing and separate
  // regions keep each 256-byte aligned for the vectorized kernel paths.
  latent_base_ = static_cast<uint8_t*>(
      arena.alloc_persistent(MemClass::DeviceHot,
                             size_t(layers) * size_t(max_token_slots_) *
                                 geo_.latent_bytes_per_token,
                             kRegionAlign));
  if (geo_.latent_scale_bytes_per_token > 0)
    latent_scale_base_ = static_cast<float*>(arena.alloc_persistent(
        MemClass::DeviceHot,
        size_t(layers) * size_t(max_token_slots_) * geo_.latent_scale_bytes_per_token,
        kRegionAlign));
  index_k_base_ = static_cast<uint8_t*>(
      arena.alloc_persistent(MemClass::DeviceHot,
                             size_t(ilayers) * size_t(pools) *
                                 geo_.index_k_bytes_per_pool,
                             kRegionAlign));
  index_scale_base_ = static_cast<float*>(arena.alloc_persistent(
      MemClass::DeviceHot, size_t(ilayers) * size_t(pools) * sizeof(float),
      kRegionAlign));
  tail_base_ = static_cast<uint8_t*>(arena.alloc_persistent(
      MemClass::DeviceHot, size_t(ilayers) * size_t(max_requests_) *
                               geo_.tail_bytes_per_request,
      kRegionAlign));
  block_tables_ = static_cast<int32_t*>(arena.alloc_persistent(
      MemClass::DeviceHot,
      size_t(max_requests_) * size_t(total_blocks_) * sizeof(int32_t),
      kRegionAlign));

  tables_host_.assign(size_t(max_requests_) * size_t(total_blocks_), 0);
  held_.assign(size_t(max_requests_), 0);
  refcount_.assign(size_t(total_blocks_), 0);
  generation_.assign(size_t(total_blocks_), 0);
  free_.reserve(size_t(total_blocks_));
  for (int64_t b = int64_t(total_blocks_) - 1; b >= 0; --b)
    free_.push_back(int32_t(b));  // LIFO: low ids come out first
  // Unheld table entries read as 0 from construction (the same invariant
  // reset_all re-establishes): device consumers that honor `held`/visible
  // bounds never touch them, and any tooling that reads the full row sees
  // deterministic zeros, not arena garbage. Synchronous memset — init is
  // cold-path and stream-less by design.
  DGPP_CUDA_OK(cudaMemset(
      block_tables_, 0,
      size_t(max_requests_) * size_t(total_blocks_) * sizeof(int32_t)));
  initialized_ = true;
}

int64_t DsaStatePool::blocks_in_use() const {
  return total_blocks_ - int64_t(free_.size());
}

int64_t DsaStatePool::block_count_for_tokens(int64_t tokens) const {
  return (tokens + cfg_.block_tokens - 1) / cfg_.block_tokens;
}

void* DsaStatePool::latent(int layer) {
  if (layer < 0 || layer >= cfg_.num_dsa_layers)
    throw std::out_of_range("dsa state pool: layer " + std::to_string(layer));
  return latent_base_ + size_t(layer) * size_t(max_token_slots_) *
                            geo_.latent_bytes_per_token;
}

float* DsaStatePool::latent_scale(int layer) {
  if (layer < 0 || layer >= cfg_.num_dsa_layers)
    throw std::out_of_range("dsa state pool: layer " + std::to_string(layer));
  if (latent_scale_base_ == nullptr) return nullptr;  // bf16 rows carry none
  return latent_scale_base_ + size_t(layer) * size_t(max_token_slots_);
}

int DsaStatePool::index_ordinal(int layer) const {
  if (layer < 0 || layer >= cfg_.num_dsa_layers)
    throw std::out_of_range("dsa state pool: layer " + std::to_string(layer));
  return index_ordinal_of(layer);
}

namespace {
int require_index_ordinal(const DsaStatePool& pool, int layer) {
  const int o = pool.index_ordinal(layer);
  if (o < 0)
    throw std::out_of_range("dsa state pool: layer " + std::to_string(layer) +
                            " owns no index cache (a shared-selection layer)");
  return o;
}
}  // namespace

void* DsaStatePool::index_k(int layer) {
  const int o = require_index_ordinal(*this, layer);
  return index_k_base_ + size_t(o) * size_t(max_pool_slots_) *
                             geo_.index_k_bytes_per_pool;
}

float* DsaStatePool::index_scale(int layer) {
  const int o = require_index_ordinal(*this, layer);
  return index_scale_base_ + size_t(o) * size_t(max_pool_slots_);
}

void* DsaStatePool::tail(int layer) {
  const int o = require_index_ordinal(*this, layer);
  return tail_base_ + size_t(o) * size_t(max_requests_) *
                          geo_.tail_bytes_per_request;
}

const void* DsaStatePool::latent(int layer) const {
  return const_cast<DsaStatePool*>(this)->latent(layer);
}
const float* DsaStatePool::latent_scale(int layer) const {
  return const_cast<DsaStatePool*>(this)->latent_scale(layer);
}
const void* DsaStatePool::index_k(int layer) const {
  return const_cast<DsaStatePool*>(this)->index_k(layer);
}
const float* DsaStatePool::index_scale(int layer) const {
  return const_cast<DsaStatePool*>(this)->index_scale(layer);
}
const void* DsaStatePool::tail(int layer) const {
  return const_cast<DsaStatePool*>(this)->tail(layer);
}

bool DsaStatePool::ensure_request_blocks(int req, int64_t tokens,
                                         cudaStream_t stream) {
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("dsa state pool: request " + std::to_string(req));
  if (tokens < 0)
    throw std::invalid_argument("dsa state pool: negative token count");
  const int64_t needed = block_count_for_tokens(tokens);
  if (needed > total_blocks_) return false;  // beyond pool capacity
  const int64_t have = held_[size_t(req)];
  if (needed <= have) return true;
  const int64_t extra = needed - have;
  if (int64_t(free_.size()) < extra) return false;  // transactional: no partial claims

  int32_t* row = tables_host_.data() + size_t(req) * size_t(total_blocks_);
  for (int64_t i = 0; i < extra; ++i) {
    row[have + i] = free_.back();
    free_.pop_back();
    refcount_[size_t(row[have + i])] = 1;
    ++generation_[size_t(row[have + i])];
  }
  held_[size_t(req)] = int32_t(needed);
  DGPP_CUDA_OK(cudaMemcpyAsync(block_tables_ + size_t(req) * size_t(total_blocks_) +
                                   size_t(have),
                               row + have, size_t(extra) * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream));
  return true;
}

void DsaStatePool::release_request_blocks(int req, cudaStream_t stream) {
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("dsa state pool: request " + std::to_string(req));
  const int64_t held = held_[size_t(req)];
  if (held == 0) return;
  int32_t* row = tables_host_.data() + size_t(req) * size_t(total_blocks_);
  for (int64_t i = held - 1; i >= 0; --i) {  // LIFO for the freed ones
    const int32_t b = row[i];
    if (--refcount_[size_t(b)] == 0) free_.push_back(b);
  }
  std::fill(row, row + held, 0);
  held_[size_t(req)] = 0;
  DGPP_CUDA_OK(cudaMemcpyAsync(
      block_tables_ + size_t(req) * size_t(total_blocks_), row,
      size_t(held) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
}

bool DsaStatePool::share_blocks_into(int req, const int32_t* blocks, int64_t n,
                                     cudaStream_t stream) {
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("dsa state pool: request " + std::to_string(req));
  if (held_[size_t(req)] != 0)
    throw std::logic_error("dsa state pool: share_blocks_into needs a fresh row");
  if (n < 0 || n > total_blocks_) return false;
  if (n == 0) return true;
  int32_t* row = tables_host_.data() + size_t(req) * size_t(total_blocks_);
  for (int64_t i = 0; i < n; ++i) {
    const int32_t b = blocks[i];
    if (b < 0 || b >= total_blocks_ || refcount_[size_t(b)] <= 0)
      throw std::logic_error("dsa state pool: sharing a block that is not live");
    row[i] = b;
    ++refcount_[size_t(b)];
  }
  held_[size_t(req)] = int32_t(n);
  DGPP_CUDA_OK(cudaMemcpyAsync(block_tables_ + size_t(req) * size_t(total_blocks_),
                               row, size_t(n) * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream));
  return true;
}

void DsaStatePool::pin_blocks(const int32_t* blocks, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    const int32_t b = blocks[i];
    if (b < 0 || b >= total_blocks_ || refcount_[size_t(b)] <= 0)
      throw std::logic_error("dsa state pool: pinning a block that is not live");
    ++refcount_[size_t(b)];
  }
}

void DsaStatePool::unpin_blocks(const int32_t* blocks, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    const int32_t b = blocks[i];
    if (b < 0 || b >= total_blocks_ || refcount_[size_t(b)] <= 0)
      throw std::logic_error("dsa state pool: unpinning a block that is not live");
    if (--refcount_[size_t(b)] == 0) free_.push_back(b);
  }
}

int32_t DsaStatePool::acquire_pinned_block() {
  if (free_.empty()) return -1;
  const int32_t b = free_.back();
  free_.pop_back();
  refcount_[size_t(b)] = 1;
  ++generation_[size_t(b)];
  return b;
}

uint64_t DsaStatePool::block_identity(int32_t block) const {
  if (block < 0 || block >= total_blocks_)
    throw std::out_of_range("dsa state pool: block index out of range");
  return (generation_[size_t(block)] << 32) | uint32_t(block);
}

std::vector<CachePlane> DsaStatePool::planes() const {
  // The same regions copy_block_contents walks, one plane per layer (and
  // per index ordinal), each with its block's bytes.
  std::vector<CachePlane> out;
  if (!initialized_) return out;
  const size_t latent_blk = size_t(cfg_.block_tokens) * geo_.latent_bytes_per_token;
  const size_t pools = size_t(geo_.pools_per_block);
  for (int layer = 0; layer < cfg_.num_dsa_layers; ++layer) {
    out.push_back({latent_base_ + size_t(layer) * size_t(max_token_slots_) * geo_.latent_bytes_per_token,
                   latent_blk});
    if (latent_scale_base_ != nullptr)
      out.push_back({reinterpret_cast<uint8_t*>(latent_scale_base_ + size_t(layer) * size_t(max_token_slots_)),
                     size_t(cfg_.block_tokens) * sizeof(float)});
  }
  for (int o = 0; o < geo_.index_layers; ++o) {
    out.push_back({index_k_base_ + size_t(o) * size_t(max_pool_slots_) * geo_.index_k_bytes_per_pool,
                   pools * geo_.index_k_bytes_per_pool});
    out.push_back({reinterpret_cast<uint8_t*>(index_scale_base_ + size_t(o) * size_t(max_pool_slots_)),
                   pools * sizeof(float)});
  }
  return out;
}

void DsaStatePool::copy_block_contents(int32_t src, int32_t dst,
                                       cudaStream_t stream) {
  if (src < 0 || src >= total_blocks_ || dst < 0 || dst >= total_blocks_)
    throw std::out_of_range("dsa state pool: block index out of range");
  if (src == dst) return;
  const size_t latent_blk = size_t(cfg_.block_tokens) * geo_.latent_bytes_per_token;
  const size_t pools = size_t(geo_.pools_per_block);
  const size_t k_blk = pools * geo_.index_k_bytes_per_pool;
  for (int layer = 0; layer < cfg_.num_dsa_layers; ++layer) {
    uint8_t* lat = latent_base_ + size_t(layer) * size_t(max_token_slots_) *
                                      geo_.latent_bytes_per_token;
    DGPP_CUDA_OK(cudaMemcpyAsync(lat + size_t(dst) * latent_blk,
                                 lat + size_t(src) * latent_blk, latent_blk,
                                 cudaMemcpyDeviceToDevice, stream));
    if (latent_scale_base_ != nullptr) {
      float* ls = latent_scale_base_ + size_t(layer) * size_t(max_token_slots_);
      const size_t bt = size_t(cfg_.block_tokens);
      DGPP_CUDA_OK(cudaMemcpyAsync(ls + size_t(dst) * bt, ls + size_t(src) * bt,
                                   bt * sizeof(float), cudaMemcpyDeviceToDevice,
                                   stream));
    }
  }
  for (int o = 0; o < geo_.index_layers; ++o) {
    uint8_t* k = index_k_base_ + size_t(o) * size_t(max_pool_slots_) *
                                     geo_.index_k_bytes_per_pool;
    DGPP_CUDA_OK(cudaMemcpyAsync(k + size_t(dst) * k_blk, k + size_t(src) * k_blk,
                                 k_blk, cudaMemcpyDeviceToDevice, stream));
    float* sc = index_scale_base_ + size_t(o) * size_t(max_pool_slots_);
    DGPP_CUDA_OK(cudaMemcpyAsync(sc + size_t(dst) * pools, sc + size_t(src) * pools,
                                 pools * sizeof(float), cudaMemcpyDeviceToDevice,
                                 stream));
  }
}

const int32_t* DsaStatePool::request_table_row(int req) const {
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("dsa state pool: request " + std::to_string(req));
  return tables_host_.data() + size_t(req) * size_t(total_blocks_);
}

int64_t DsaStatePool::request_blocks(int req) const {
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("dsa state pool: request " + std::to_string(req));
  return held_[size_t(req)];
}

void DsaStatePool::reset_all(cudaStream_t stream) {
  if (!initialized_)
    throw std::logic_error("dsa state pool: not initialized");
  const size_t layers = size_t(cfg_.num_dsa_layers);
  const size_t ilayers = size_t(geo_.index_layers);
  DGPP_CUDA_OK(cudaMemsetAsync(
      latent_base_, 0,
      layers * size_t(max_token_slots_) * geo_.latent_bytes_per_token, stream));
  if (latent_scale_base_ != nullptr)
    DGPP_CUDA_OK(cudaMemsetAsync(latent_scale_base_, 0,
                                 layers * size_t(max_token_slots_) * sizeof(float),
                                 stream));
  DGPP_CUDA_OK(cudaMemsetAsync(
      index_k_base_, 0,
      ilayers * size_t(max_pool_slots_) * geo_.index_k_bytes_per_pool, stream));
  DGPP_CUDA_OK(cudaMemsetAsync(
      index_scale_base_, 0, ilayers * size_t(max_pool_slots_) * sizeof(float),
      stream));
  DGPP_CUDA_OK(cudaMemsetAsync(
      tail_base_, 0,
      ilayers * size_t(max_requests_) * geo_.tail_bytes_per_request, stream));

  // Return every block to the free list and zero the whole table (rows AND
  // the unheld tail of each row, so a stale device read fails against a
  // deterministically-zero table rather than a dangling id).
  std::fill(tables_host_.begin(), tables_host_.end(), 0);
  std::fill(held_.begin(), held_.end(), 0);
  free_.clear();
  std::fill(refcount_.begin(), refcount_.end(), 0);
  free_.reserve(size_t(total_blocks_));
  for (int64_t b = int64_t(total_blocks_) - 1; b >= 0; --b)
    free_.push_back(int32_t(b));
  DGPP_CUDA_OK(cudaMemsetAsync(
      block_tables_, 0,
      size_t(max_requests_) * size_t(total_blocks_) * sizeof(int32_t),
      stream));
}

void DsaStatePool::reset_request(int req, cudaStream_t stream) {
  if (req < 0 || req >= max_requests_)
    throw std::out_of_range("dsa state pool: request " + std::to_string(req));
  // The tail region is layer-major with max_requests slots per layer, so a
  // request's rings are strided across layers — one small stream-ordered
  // memset per layer, once per request open. The ring is the only cache a
  // fresh request can read before writing (the tail-seed read), so it is
  // the only one reset_request must zero.
  for (int o = 0; o < geo_.index_layers; ++o) {
    uint8_t* slot =
        tail_base_ +
        size_t(o) * size_t(max_requests_) * geo_.tail_bytes_per_request +
        size_t(req) * geo_.tail_bytes_per_request;
    DGPP_CUDA_OK(
        cudaMemsetAsync(slot, 0, geo_.tail_bytes_per_request, stream));
  }
  release_request_blocks(req, stream);
}

size_t DsaStatePool::cache_bytes(const DsaConfig& cfg, int max_requests,
                                 int64_t max_token_slots) {
  if (max_requests <= 0 || max_token_slots <= 0 ||
      max_token_slots % cfg.block_tokens != 0)
    throw std::invalid_argument("dsa state pool: invalid accounting shape");
  const DsaGeometry g = DsaGeometry::from_config(cfg);
  const int64_t pools =
      max_token_slots / cfg.block_tokens * g.pools_per_block;
  const size_t layers = size_t(cfg.num_dsa_layers);
  const size_t ilayers = size_t(g.index_layers);
  return padded(layers * size_t(max_token_slots) * g.latent_bytes_per_token) +
         (g.latent_scale_bytes_per_token > 0
              ? padded(layers * size_t(max_token_slots) * g.latent_scale_bytes_per_token)
              : 0) +
         padded(ilayers * size_t(pools) * g.index_k_bytes_per_pool) +
         padded(ilayers * size_t(pools) * sizeof(float)) +
         padded(ilayers * size_t(max_requests) * g.tail_bytes_per_request) +
         padded(size_t(max_requests) *
                    size_t(max_token_slots / cfg.block_tokens) *
                    sizeof(int32_t));
}

}  // namespace dgpp
