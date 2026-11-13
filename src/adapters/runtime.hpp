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
  // rider (may be null): the digit rows of the boundary these rows will
  // fold at, fold_rows rows in all (rider_elems of them). The projection
  // then writes this rank's digits itself and add_from_rider completes it.
  void project(int layer, bool attention, const uint16_t* x, int stride, int row_begin, int rows,
               cudaStream_t stream, uint16_t* rider = nullptr, int fold_rows = 0);
  void add(int layer, bool attention, uint16_t* output, int rows, cudaStream_t stream,
           bool capture = false);
  // add() around a boundary collective that carries the rank-space rows
  // itself (BoundaryReducer::rider): the bf16 elements the rows need as
  // byte digits (0: no factor here, or no stream transport), which
  // project() writes for this rank, and behind the fold their sum and the
  // Bz addition in one kernel. The same values as add()'s own collective.
  // Every row of the boundary must have been projected into the rider.
  size_t rider_elems(int layer, bool attention, int rows) const;
  void add_from_rider(int layer, bool attention, const uint16_t* digits, uint16_t* output, int rows,
                      cudaStream_t stream);
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
  int rank_index_, world_;
  int rider_site_ = -1, rider_rows_ = 0;  // the boundary whose rider is being written, and its rows so far
  SumFp32 sum_;
  SumFp32Stream stream_sum_;
  std::vector<std::unique_ptr<DeviceFactor>> factors_;
  float *host_z_ = nullptr, *device_z_ = nullptr, *host_delta_ = nullptr, *device_delta_ = nullptr;
};
}  // namespace dgpp::adapters
