# GDN normalized Q/K boundary — built; four GPU tests pass; long replay pending

Relative to the FP32-router control, this adds BF16 rounding of normalized
GDN queries and keys before the recurrent update; query scaling follows that
boundary. The scalar-gate specialization alone changes, leaving KDA unchanged.
The pinned reference explicitly stores normalized prefill Q/K in BF16. Its
active fused prefill beta path uses FP32, which this candidate preserves.

This diagnostic applies that Q/K boundary to both plain and batched GDN paths
to preserve DGPP's chunk/batch/snapshot invariants. It is not an exact clone of
the reference decode kernel, which normalizes live Q/K in FP32. It also does
not implement the reference's chunk-parallel recurrent algorithm.

The independent host FP32/FP64 recurrence includes the explicit BF16 storage
boundary. All existing GDN recurrence, chunk, snapshot and batch tests must
pass before the unchanged original TP2 request. Bounded prefill follows only
if native succeeds. Production restoration runs in finally.

All four existing GDN GPU tests passed on idle node 192.168.88.13 with the
retained test executable: FP32/FP64 recurrence parity, bitwise chunk invariance,
bitwise prefix snapshots and bitwise batch/per-request equivalence. The reference
world occupied only nodes 11 and 12. The TP2 campaign verifies this receipt and
its binary identity before starting.
