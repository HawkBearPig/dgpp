#pragma once
// The DFlash2 drafter's plain bf16 K/V planes over the target pool's
// shared block table (models/dflash2/drafter.hpp): the draft's KV context
// rides the request's blocks, so sharing, copying and release need no
// protocol of their own — every block index of the target pool addresses
// these verbatim. Hosted by value inside a target KV pool; 0 layers: no
// planes.
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"

namespace dgpp {

class DraftKvPlanes {
 public:
  struct View {
    uint16_t* k = nullptr;
    uint16_t* v = nullptr;
  };

  DraftKvPlanes() = default;
  DraftKvPlanes(const DraftKvPlanes&) = delete;
  DraftKvPlanes& operator=(const DraftKvPlanes&) = delete;
  DraftKvPlanes(DraftKvPlanes&& o) noexcept { *this = std::move(o); }
  DraftKvPlanes& operator=(DraftKvPlanes&& o) noexcept {
    if (this != &o) {
      free();
      dk_ = o.dk_;
      dv_ = o.dv_;
      layers_ = o.layers_;
      kv_heads_ = o.kv_heads_;
      dim_ = o.dim_;
      token_slots_ = o.token_slots_;
      block_tokens_ = o.block_tokens_;
      o.dk_ = nullptr;
      o.dv_ = nullptr;
      o.layers_ = 0;
    }
    return *this;
  }
  ~DraftKvPlanes() { free(); }

  // The device bytes for a shape (the pool's memory plan): k and v, bf16,
  // per draft layer.
  static size_t bytes(int layers, int kv_heads, int dim, int64_t token_slots) {
    return layers > 0 ? 2 * static_cast<size_t>(layers) * static_cast<size_t>(token_slots) *
                            static_cast<size_t>(kv_heads) * static_cast<size_t>(dim) * 2
                      : 0;
  }
  size_t bytes() const { return bytes(layers_, kv_heads_, dim_, token_slots_); }
  int layers() const { return layers_; }
  bool empty() const { return layers_ == 0; }

  void init(int layers, int kv_heads, int dim, int64_t token_slots, int block_tokens) {
    free();
    if (layers < 0 || kv_heads < 0 || dim < 0 || token_slots < 0 || block_tokens < 0 ||
        (layers > 0 && (kv_heads <= 0 || dim <= 0)))
      throw std::invalid_argument("DraftKvPlanes: planes need positive geometry");
    layers_ = layers;
    kv_heads_ = kv_heads;
    dim_ = dim;
    token_slots_ = token_slots;
    block_tokens_ = block_tokens;
    if (layers > 0) {
      DGPP_CUDA_OK(cudaMalloc(&dk_, bytes() / 2));
      DGPP_CUDA_OK(cudaMalloc(&dv_, bytes() / 2));
    }
  }
  void free() {
    cudaFree(dk_);
    cudaFree(dv_);
    dk_ = nullptr;
    dv_ = nullptr;
    layers_ = 0;
  }

  // The kernels' view of draft layer `l` (the DRAFT stack's index, not the
  // target's).
  View view(int l) const {
    if (l < 0 || l >= layers_)
      throw std::out_of_range("DraftKvPlanes: draft layer " + std::to_string(l));
    const size_t row = static_cast<size_t>(kv_heads_) * dim_;
    return {dk_ + static_cast<size_t>(l) * static_cast<size_t>(token_slots_) * row,
            dv_ + static_cast<size_t>(l) * static_cast<size_t>(token_slots_) * row};
  }
  // Every layer's rows of physical block `src` into `dst`, stream-ordered
  // (the pool's copy protocol).
  void copy_block(int32_t src, int32_t dst, cudaStream_t stream) const {
    if (layers_ == 0) return;
    const size_t row = static_cast<size_t>(kv_heads_) * dim_;
    const size_t span = static_cast<size_t>(block_tokens_) * row * 2;
    for (int l = 0; l < layers_; ++l) {
      const View v = view(l);
      DGPP_CUDA_OK(cudaMemcpyAsync(v.k + static_cast<size_t>(dst) * row, v.k + static_cast<size_t>(src) * row,
                                   span, cudaMemcpyDeviceToDevice, stream));
      DGPP_CUDA_OK(cudaMemcpyAsync(v.v + static_cast<size_t>(dst) * row, v.v + static_cast<size_t>(src) * row,
                                   span, cudaMemcpyDeviceToDevice, stream));
    }
  }
  // Cold start / snapshot restore: zero every plane (the table rows are the
  // pool's).
  void reset(cudaStream_t stream) const {
    if (layers_ > 0) {
      DGPP_CUDA_OK(cudaMemsetAsync(dk_, 0, bytes() / 2, stream));
      DGPP_CUDA_OK(cudaMemsetAsync(dv_, 0, bytes() / 2, stream));
    }
  }

 private:
  int layers_ = 0;
  int kv_heads_ = 0;
  int dim_ = 0;
  int64_t token_slots_ = 0;
  int block_tokens_ = 0;
  uint16_t* dk_ = nullptr;  // [layers][token_slots][kv_heads*dim] bf16
  uint16_t* dv_ = nullptr;
};

}  // namespace dgpp
