# DGPP W4A16 with closer reference rounding — 5/6, original wrong answer

This diagnostic removes activation FP4 rounding and restores the original
weight-only global scale from the prior post-dot arithmetic combination. It
retains trained EOS preservation, FP32 GR intermediates, single-rounding SwiGLU,
checkpoint BF16 dense projections and complete resident-image bypass.

The motivation is the pinned Marlin W4A16 reference's two identical 6/6 results on the
unchanged request. The prior closer W4A4 DGPP
combination matched the deterministic W4A4 reference's entire wrong answer.

This is not an exact Marlin arithmetic clone. DGPP still rounds routing weights
to BF16 and sums expert down products in FP32 before its final BF16 rounding.
Marlin's expert weighting/staging differs. Other reduction orders and GDN
intermediates also remain implementation-specific.

Native original request runs first; bounded prefill runs only if native passes.
Any success needs repetition, ablation and broader numerical-quality validation
before promoting an arithmetic policy change.

Executed result: 5/6, `val_5dac9ed720abddaf`, 184 completion tokens, 209.544
seconds. Bounded prefill was skipped after the native failure. Production binary,
configuration and inference restoration passed on all four nodes.
