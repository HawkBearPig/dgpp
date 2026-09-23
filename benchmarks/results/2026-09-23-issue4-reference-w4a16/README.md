# Matched activation-precision reference — running

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
