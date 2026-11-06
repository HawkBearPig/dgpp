#pragma once
#include <functional>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "adapters/artifact.hpp"

namespace dgpp::adapters {
// The owner supplies the optional FP32 rank-space fold. Single-node execution
// needs no transport. Construction/installation requires explicit opt-in.
using SumFp32 = std::function<void(float*, size_t)>;
// Stream-ordered FP32 fold, including CUDA graph recording. The callback owns
// its stable transport scratch; capture must enqueue only device operations.
using SumFp32Stream = std::function<void(float*, size_t, cudaStream_t, bool)>;
class Runtime {
 public:
  Runtime(const Manifest& manifest, int rank, int world, int max_rows, int hidden, int layers,
          SumFp32 sum, bool enabled, SumFp32Stream stream_sum = {});
  ~Runtime();
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;
  bool has(int layer, bool attention) const;
  void project(int layer, bool attention, const uint16_t* x, int stride, int row_begin, int rows,
               cudaStream_t stream);
  void add(int layer, bool attention, uint16_t* output, int rows, cudaStream_t stream,
           bool capture = false);
  // Turn shared-expert local captures into the represented global output,
  // including the correction. Only the explicit research walk calls this.
  void shared_capture(int layer, std::vector<float>& output, int rows, cudaStream_t stream);
  const std::string& id() const { return id_; }
  static size_t factor_bytes(const Manifest& manifest, int world);
  static size_t serving_scratch_bytes(const Manifest& manifest, int world, int max_rows,
                                      int hidden);
  static int extra_graph_nodes(const Manifest& manifest, int world, int hidden);

 private:
  struct DeviceFactor;
  const DeviceFactor* factor(int layer, bool attention) const;
  std::string id_;
  int max_rows_, hidden_;
  SumFp32 sum_;
  SumFp32Stream stream_sum_;
  std::vector<std::unique_ptr<DeviceFactor>> factors_;
  float *host_z_ = nullptr, *device_z_ = nullptr, *host_delta_ = nullptr, *device_delta_ = nullptr;
};
}  // namespace dgpp::adapters
