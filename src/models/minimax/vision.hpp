#pragma once
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"
#include "models/minimax/vision_config.hpp"

namespace dgpp {
class MinimaxVisionEncoder {
 public:
  using Trace = std::function<void(const std::string&, const void*, size_t, DType)>;
  MinimaxVisionEncoder(const MinimaxVisionConfig&, const std::string& checkpoint, cudaStream_t);
  ~MinimaxVisionEncoder();
  const uint16_t* encode(const ImageInput&, Trace = {});
  const uint16_t* stage(const std::vector<ImageInput>&, int64_t first, int64_t end);
  uint64_t digest() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace dgpp
