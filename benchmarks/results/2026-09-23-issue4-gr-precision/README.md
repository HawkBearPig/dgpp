# Issue #4: gated-residual intermediate precision diagnostic

The reporter's exact vLLM image uses NVIDIA's fused HC kernels. Its gate mix
accumulates FP32 sigmoid(gate) times BF16 normalized residuals in FP32, then
rounds the mean to BF16. Its injection computes the sigmoid and multiply/add in
FP32 and rounds the updated residual once. DGPP additionally rounds sigmoid
outputs and individual products to BF16. This is an actual arithmetic-policy
difference, not an established indexing bug.

The diagnostic changes only these intermediate roundings and uses FP32 FMA,
leaving baseline EOS, W4A16 experts and FP8 dense settings intact. A separate
focused executable compares mix, combine, injection gates and fused injection
paths with an independent FP64 mathematical oracle at the real 4x2560 geometry.
The focused test establishes whether each implementation follows this
higher-precision contract. It does not by itself establish which policy is
correct for the checkpoint. The causal test replays the original TP2 request.

The exact reference image sources are retained in the adjacent issue4-reference
record. This patch remains diagnostic pending original-request and broader
quality validation; the old BF16-staged contract is intentional in existing tests.

The original-request run still returns 5/6, with the baseline wrong value and
184 completion tokens. This arithmetic difference does not fix the issue. The
candidate focused test passed (mix L2 1.417e-5, combine L2 6.340e-6, gates about
4.4e-8 over 15,449,600 scalar outputs). The first standalone baseline build
omitted nvcc relaxed-constexpr support, emitted warnings and produced zero
outputs; its apparent failure is invalid evidence. It is retained explicitly
as invalid, rebuilt with the repository's CUDA flags, and rerun serially. The corrected baseline has relative L2 0.003062 for mix,
0.002176 for combine, and 0.00163 for gates. The candidate remains below the
focused tolerance. The candidate test and server used the proper CMake build.
The bounded original-request run was skipped after the native result remained
wrong. Both campaigns independently verified production restoration.
