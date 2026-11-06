#include "adapters/runtime.hpp"

#include <algorithm>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "kernels/low_rank.hpp"

namespace dgpp::adapters {
struct Runtime::DeviceFactor {
  float *a = nullptr, *b = nullptr;
  int rank = 0, inputs = 0;
  ~DeviceFactor() {
    cudaFree(a);
    cudaFree(b);
  }
};
size_t Runtime::factor_bytes(const Manifest& manifest, int world) {
  size_t bytes = 0;
  for (const auto& p : manifest.projections)
    bytes += p.rank * (p.target.input_features / world + p.target.output_features) * sizeof(float);
  return bytes;
}
Runtime::Runtime(const Manifest& manifest, int rank, int world, int max_rows, int hidden,
                 int layers, SumFp32 sum, bool enabled)
    : id_(manifest.id), max_rows_(max_rows), hidden_(hidden), sum_(std::move(sum)) {
  if (!enabled)
    throw std::invalid_argument("runtime adapters require engine.runtime_adapters=true");
  if (rank < 0 || rank >= world || manifest.world != world || max_rows < 1 || max_rows > 2048 ||
      hidden < 1 || layers < 1 || (world > 1 && !sum_))
    throw std::invalid_argument("invalid runtime adapter geometry/transport");
  factors_.resize(static_cast<size_t>(layers) * 2);
  int max_rank = 1;
  for (const auto& p : manifest.projections) {
    if (p.a.size() != static_cast<size_t>(p.rank) * p.target.input_features ||
        p.b.size() != static_cast<size_t>(p.rank) * hidden)
      throw std::invalid_argument("runtime adapter factors are incomplete");
    // A zero factor is an exact identity: skip even the BF16 addition.
    if (std::all_of(p.a.begin(), p.a.end(), [](float v) { return v == 0; }) ||
        std::all_of(p.b.begin(), p.b.end(), [](float v) { return v == 0; }))
      continue;
    auto f = std::make_unique<DeviceFactor>();
    f->rank = p.rank;
    f->inputs = p.target.input_features / world;
    std::vector<float> local_a(static_cast<size_t>(f->rank) * f->inputs);
    for (int r = 0; r < f->rank; ++r)
      std::copy_n(p.a.data() + static_cast<size_t>(r) * p.target.input_features + rank * f->inputs,
                  f->inputs, local_a.data() + static_cast<size_t>(r) * f->inputs);
    DGPP_CUDA_OK(cudaMalloc(&f->a, local_a.size() * sizeof(float)));
    DGPP_CUDA_OK(cudaMalloc(&f->b, p.b.size() * sizeof(float)));
    DGPP_CUDA_OK(
        cudaMemcpy(f->a, local_a.data(), local_a.size() * sizeof(float), cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMemcpy(f->b, p.b.data(), p.b.size() * sizeof(float), cudaMemcpyHostToDevice));
    max_rank = std::max(max_rank, f->rank);
    factors_.at(static_cast<size_t>(p.target.layer) * 2 +
                (p.target.component == "attn.o_proj" ? 0 : 1)) = std::move(f);
  }
  try {
    DGPP_CUDA_OK(cudaHostAlloc(&host_z_, static_cast<size_t>(max_rows) * max_rank * sizeof(float),
                               cudaHostAllocMapped));
    DGPP_CUDA_OK(cudaHostGetDevicePointer(&device_z_, host_z_, 0));
    DGPP_CUDA_OK(cudaHostAlloc(&host_delta_, hidden * sizeof(float), cudaHostAllocMapped));
    DGPP_CUDA_OK(cudaHostGetDevicePointer(&device_delta_, host_delta_, 0));
  } catch (...) {
    cudaFreeHost(host_z_);
    cudaFreeHost(host_delta_);
    throw;
  }
}
Runtime::~Runtime() {
  cudaFreeHost(host_z_);
  cudaFreeHost(host_delta_);
}
const Runtime::DeviceFactor* Runtime::factor(int layer, bool attention) const {
  return factors_.at(static_cast<size_t>(layer) * 2 + (attention ? 0 : 1)).get();
}
bool Runtime::has(int layer, bool attention) const {
  return factor(layer, attention) != nullptr;
}
void Runtime::project(int layer, bool attention, const uint16_t* x, int stride, int row_begin,
                      int rows, cudaStream_t stream) {
  const auto* f = factor(layer, attention);
  if (!f) return;
  if (row_begin < 0 || rows <= 0 || row_begin > max_rows_ - rows || stride < f->inputs)
    throw std::invalid_argument("runtime adapter row/input bounds");
  low_rank_project(x, stride, f->a, device_z_ + static_cast<size_t>(row_begin) * f->rank, rows,
                   f->inputs, f->rank, stream);
}
void Runtime::add(int layer, bool attention, uint16_t* output, int rows, cudaStream_t stream) {
  const auto* f = factor(layer, attention);
  if (!f) return;
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  if (sum_) sum_(host_z_, static_cast<size_t>(rows) * f->rank);
  low_rank_add(device_z_, f->b, output, rows, hidden_, f->rank, stream);
}
void Runtime::shared_capture(int layer, std::vector<float>& output, int rows, cudaStream_t stream) {
  if (output.size() != static_cast<size_t>(hidden_))
    throw std::invalid_argument("shared capture shape");
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  if (sum_) sum_(output.data(), output.size());
  if (const auto* f = factor(layer, false)) {
    low_rank_delta_last(device_z_, f->b, device_delta_, rows, hidden_, f->rank, stream);
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    for (int c = 0; c < hidden_; ++c) output[c] += host_delta_[c];
  }
}
}  // namespace dgpp::adapters
