// The decode graph's programmatic edges (2026-10-20). A captured decode
// pass is a chain of ~1,750 kernel nodes a layer apart by ~0.5 us of
// launch latency but each with its own ramp and drain: the node trace of
// the C1 pass put the dense GEMVs 4-22 % under their isolated rates for
// that reason alone (isolated, back-to-back launches overlap). A
// programmatic edge lets a kernel's blocks be scheduled as its
// predecessor's drain, each waiting (griddepcontrol.wait, kernels/pdl.cuh)
// before its first read; the probe measured 1.2 us a transition recovered
// on this GPU. Every kernel of ours carries the wait (the source audit
// holds it: a kernel without it, converted, would read its predecessor's
// output early — the Qwen and MiMo transcripts did, 2026-10-20, when a
// name registry matched kernels it had not patched), so every dependent
// in our namespace takes the edge and a library kernel keeps a plain one.
// Arithmetic unchanged: bitwise.
#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>
#include "common/cuda_check.hpp"
#include "common/log.hpp"

namespace dgpp {
namespace pdl {
// Every kernel in the tree carries the wait as its first statement (the
// source audit tests/scripts/pdl_wait_audit.py holds it), so any kernel in
// our namespace takes the edge; a library's kernel (cuBLASLt's) keeps a
// plain edge.
inline bool ours(const char* name) { return name != nullptr && std::strncmp(name, "_ZN4dgpp", 8) == 0; }

inline bool ready(const char* name) { return ours(name); }

struct EdgeStats {
  int kernel_nodes = 0, edges = 0, converted = 0;
  std::vector<std::string> unlisted;  // kernel names left on plain edges (distinct)
};

// Rewrites every kernel -> kernel dependency whose dependent carries the
// wait into a programmatic edge (the dependent launches as the primary's
// blocks exit). Non-kernel nodes and unlisted kernels keep plain edges.
inline EdgeStats programmatic_edges(cudaGraph_t graph) {
  EdgeStats st;
  size_t n = 0;
  DGPP_CUDA_OK(cudaGraphGetNodes(graph, nullptr, &n));
  std::vector<cudaGraphNode_t> nodes(n);
  DGPP_CUDA_OK(cudaGraphGetNodes(graph, nodes.data(), &n));
  auto kernel_name = [](cudaGraphNode_t node, const char** name) {
    cudaGraphNodeType type;
    DGPP_CUDA_OK(cudaGraphNodeGetType(node, &type));
    if (type != cudaGraphNodeTypeKernel) return false;
    cudaKernelNodeParams p{};
    if (cudaGraphKernelNodeGetParams(node, &p) != cudaSuccess || p.func == nullptr) { (void)cudaGetLastError(); return false; }
    if (cudaFuncGetName(name, p.func) != cudaSuccess) { (void)cudaGetLastError(); *name = nullptr; }
    return true;
  };
  for (cudaGraphNode_t node : nodes) {
    const char* name = nullptr;
    if (!kernel_name(node, &name)) continue;
    ++st.kernel_nodes;
    size_t nd = 0;
    DGPP_CUDA_OK(cudaGraphNodeGetDependencies(node, nullptr, nullptr, &nd));
    if (nd == 0) continue;
    std::vector<cudaGraphNode_t> deps(nd);
    std::vector<cudaGraphEdgeData> data(nd);
    DGPP_CUDA_OK(cudaGraphNodeGetDependencies(node, deps.data(), data.data(), &nd));
    st.edges += static_cast<int>(nd);
    if (!ready(name)) {
      std::string s = name ? name : "?";
      if (std::find(st.unlisted.begin(), st.unlisted.end(), s) == st.unlisted.end()) st.unlisted.push_back(s);
      continue;
    }
    for (size_t i = 0; i < nd; ++i) {
      const char* from_name = nullptr;
      if (!kernel_name(deps[i], &from_name)) continue;  // only kernel -> kernel edges
      if (data[i].type != cudaGraphDependencyTypeDefault) continue;
      cudaGraphEdgeData e{};
      e.from_port = cudaGraphKernelNodePortProgrammatic;
      e.to_port = 0;
      e.type = cudaGraphDependencyTypeProgrammatic;
      DGPP_CUDA_OK(cudaGraphRemoveDependencies(graph, &deps[i], &node, &data[i], 1));
      DGPP_CUDA_OK(cudaGraphAddDependencies(graph, &deps[i], &node, &e, 1));
      ++st.converted;
    }
  }
  return st;
}
}  // namespace pdl
}  // namespace dgpp
