# Matched activation-precision reference — 6/6 twice

Both unchanged original requests return all six correct associations. The
assistant text and usage are identical: 261120 prompt tokens, 187 completion
tokens and normal stop. All input IDs match DGPP; prefix caching is disabled.
Wall times are 292.516 and 293.350 seconds. Both ranks select MARLIN.

This is repeatability in one two-request world, not universal determinism.
Compared with the prior deterministic W4A4 reference, this also omits tracing
and read-only top-k probes. It is not yet a one-variable causal proof that
activation quantization alone causes the error.

The pinned reference supports a Marlin NVFP4 W4A16 expert backend. This removes
the W4A4-versus-W4A16 activation precision difference with DGPP. It retains the
exact deterministic QSA selector already independently tested, without tracing
or read-only top-k probes. The original request runs twice in one fresh world.

Source inspection confirms Marlin disables activation quantization, uses FP32
GEMM reduction with atomic addition disabled, and supports the checkpoint's
SwiGLU clamp. It is still a separate implementation and other rounding/reduction
differences remain. A result here cannot alone establish DGPP correctness.

Image, checkpoint, TP2, original token sequence and request settings are pinned
as before. Production restoration and verification execute in `finally`.

Production restoration completed and passed original binary/configuration and
inference checks on all four nodes.
