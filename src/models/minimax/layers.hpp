#pragma once
#include <memory>

#include "common/paged_attention_rows.hpp"
#include "models/minimax/kv_pool.hpp"
#include "models/minimax/loader.hpp"

namespace dgpp {
class MinimaxAttentionLayer {
 public:
  MinimaxAttentionLayer(const MinimaxConfig&, const MinimaxLocalGeometry&, int max_tokens,
                        int64_t cache_tokens);
  ~MinimaxAttentionLayer();
  void enqueue(const MinimaxAttnResident&, const uint16_t* x, int tokens, const PagedAttentionRows&,
               const MinimaxKvCache&, uint16_t* out, cudaStream_t);
  static size_t scratch_bytes(const MinimaxConfig&, const MinimaxLocalGeometry&, int max_tokens,
                              int64_t cache_tokens);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
class MinimaxMlp {
 public:
  MinimaxMlp(const MinimaxConfig&, int local_inter, int max_tokens);
  ~MinimaxMlp();
  void enqueue(const MinimaxMlpResident&, const uint16_t* x, int tokens, uint16_t* out,
               cudaStream_t);
  static size_t scratch_bytes(int local_inter, int max_tokens);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
class MinimaxMoeLayer {
 public:
  MinimaxMoeLayer(const MinimaxConfig&, const MinimaxLocalGeometry&, int max_tokens);
  ~MinimaxMoeLayer();
  void prepare_table(int layer, const MinimaxMoeResident&, cudaStream_t);
  void enqueue(int layer, const MinimaxMoeResident&, const uint16_t* x, int tokens, uint16_t* out,
               cudaStream_t);
  const int32_t* route_ids() const;
  const float* route_weights() const;
  static size_t scratch_bytes(const MinimaxConfig&, const MinimaxLocalGeometry&, int max_tokens);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace dgpp
