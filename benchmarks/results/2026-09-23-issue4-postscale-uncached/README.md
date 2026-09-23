# Uncached post-dot arithmetic control — executed, 5/6

Rebuilt every weight directly from the original checkpoint. This corrects the
invalid resident-image integration in the adjacent postscale control. The
arithmetic remains a diagnostic combination, not an established fix: native
activation block quantization with global input/weight scaling after the dot,
single-rounding SwiGLU and FP32 gated-residual intermediates. Dense projections
use checkpoint BF16 weights. BF16 tensor-core accumulation and DGPP expert
reduction remain distinct from the reference.

`diagnostic.patch` plus `issue4_input_scales.hpp` on the identified base reproduce
the retained executable. The native original request runs first; bounded prefill
runs only if native passes. Production restoration executes in `finally`.

The unchanged request returns `val_5b9d2c3d7a95b986` for the failing key,
with 261120 prompt tokens, 190 output tokens, zero cached tokens and normal stop.
Both rank logs confirm that resident images were bypassed. The prior malformed
response was therefore an invalid scale-integration control, not the behavior
of this corrected arithmetic combination. Bounded prefill was skipped after
the native accuracy failure.

The assistant text and token counts match the deterministic reference exactly
(`reference-parity.json`). This is an output-level observation, not a claim of
identical hidden activations. The combination still does not fix #4 and remains
diagnostic.

Production restoration completed and passed binary/configuration/inference checks.
