#pragma once
#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"
#include "common/log.hpp"

namespace dgpp {

// CUDA Graph manager keyed by step-shape class (DESIGN §11). Capture runs
// the builder once against a capture-mode stream; replay instantiates once
// and relaunches. Builders must be side-effect free on model state because
// capture does not execute kernels.
class GraphCache {
 public:
  using Key = uint64_t;
  using BuildFn = std::function<void(cudaStream_t)>;

  struct Entry {
    cudaGraph_t graph{};
    cudaGraphExec_t exec{};
    std::string label;
  };

  GraphCache() = default;
  GraphCache(const GraphCache&) = delete;
  GraphCache& operator=(const GraphCache&) = delete;
  ~GraphCache() { clear(); }

  // Replays cached graph; captures + instantiates on first use of `key`.
  // The build lambda receives a fresh capture stream to record into.
  void replay_or_capture(Key key, std::string label, cudaStream_t launch_stream,
                         const BuildFn& build) {
    auto it = entries_.find(key);
    if (it != entries_.end()) {
      ++hits_;
      DGPP_CUDA_OK(cudaGraphLaunch(it->second.exec, launch_stream));
      return;
    }
    ++captures_;
    cudaGraph_t g{};
    cudaGraphExec_t gx{};
    DGPP_CUDA_OK(cudaStreamBeginCapture(launch_stream, cudaStreamCaptureModeGlobal));
    try {
      build(launch_stream);
    } catch (...) {
      // A throwing builder must not leave the stream capturing: abort the
      // capture (dropping the partial graph) so the stream stays usable
      // and the caller can fall back to eager.
      cudaGraph_t abandoned{};
      (void)cudaStreamEndCapture(launch_stream, &abandoned);
      if (abandoned != nullptr) cudaGraphDestroy(abandoned);
      throw;
    }
    DGPP_CUDA_OK(cudaStreamEndCapture(launch_stream, &g));
    DGPP_CUDA_OK(cudaGraphInstantiate(&gx, g, 0));
    entries_[key] = Entry{g, gx, std::move(label)};
    DGPP_LOG_INFO("graph captured key={:#x} label={} nodes={} (total captures={})",
                  key, entries_[key].label, node_count(g), captures_);
    // Capture records without executing: the first caller expects its work
    // to run like any other invocation, so launch right away.
    DGPP_CUDA_OK(cudaGraphLaunch(gx, launch_stream));
  }

  bool contains(Key key) const { return entries_.count(key) != 0; }

  // Launches an already-captured graph without a capture fallback.
  void launch(Key key, cudaStream_t stream) {
    auto it = entries_.find(key);
    if (it == entries_.end())
      throw std::runtime_error("graph cache: unknown key " +
                               std::to_string(key));
    ++hits_;
    DGPP_CUDA_OK(cudaGraphLaunch(it->second.exec, stream));
  }

  uint64_t hits() const { return hits_; }
  uint64_t captures() const { return captures_; }

  void clear() {
    for (auto& [k, e] : entries_) {
      if (e.exec) cudaGraphExecDestroy(e.exec);
      if (e.graph) cudaGraphDestroy(e.graph);
    }
    entries_.clear();
    hits_ = captures_ = 0;
  }

 private:
  static int node_count(cudaGraph_t g) {
    size_t n = 0;
    cudaGraphGetNodes(g, nullptr, &n);
    return static_cast<int>(n);
  }

  std::map<Key, Entry> entries_;
  uint64_t hits_ = 0;
  uint64_t captures_ = 0;
};

}  // namespace dgpp
