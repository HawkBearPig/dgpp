// Programmatic dependent launch (2026-10-20): a kernel launched behind a
// programmatic graph edge may start while its predecessor drains; it waits
// here, before its first read of the predecessor's data, for the
// predecessor grid to complete and its writes to be visible. On a plain
// edge or a plain launch the instruction returns at once, so every decode
// kernel carries it unconditionally as its first statement
// (engine/graph_pdl.hpp rewrites the decode graph's edges for the kernels
// listed there).
#pragma once
__device__ __forceinline__ void dgpp_pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
