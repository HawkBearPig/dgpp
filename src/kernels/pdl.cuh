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
// A kernel whose running time its dependents can use releases them here:
// once every block of the grid has passed this point the dependents'
// blocks are scheduled beside it and park at their own wait. The
// collective kernels release on request (a collective is a network round
// trip with the memory idle). A parked block holds its threads, registers
// and shared memory, so the release is for a predecessor whose dependent
// is known and small against the SM, never a blanket.
__device__ __forceinline__ void dgpp_pdl_release() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}
// A kernel that reads its own weights BEFORE the wait opens with this
// marker in place of the wait. Behind a released predecessor those loads
// run beside the predecessor instead of behind it. Before the wait a
// kernel may read only what no kernel of the pass writes (weights and its
// arguments) and may write only its own registers and shared memory. The
// wait still comes before its first read of anything a kernel writes
// (tests/python/pdl_wait_audit.py holds the marker and the wait behind it).
__device__ __forceinline__ void dgpp_pdl_preamble() {}
