# GDN normalization rounding control — GPU operator checks pass, original request running

The prior W4A16 GR/SwiGLU/checkpoint-dense combination still returns the original
5/6 answer. This control changes only GDN gated normalization relative to that
binary: normalized activations and the affine product stay in FP32 until the
final sigmoid-gated BF16 store, matching the pinned reference layernorm kernel.
The host reduction order remains different from the device reduction.

On 663552 real captured values across 72 rank/layer cases, 224898 rounded values
change. The proposed policy has lower FP64 error in all 72 cases; maximum
relative L2 decreases from 0.00324458 to 0.00181016. This is numerical evidence,
not yet evidence that the original retrieval failure is repaired.

The original request runs on fresh TP2; bounded prefill runs only if native passes.
The campaign restores and verifies the original production world in `finally`.
If successful, further ablation must establish the minimal necessary fix.

All three focused GPU normalization tests passed, including the independent
FP64 GDN oracle and a check that distinguishes the old rounding policy.
