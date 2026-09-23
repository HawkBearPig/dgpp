# FP32 expert routing weights — building, not yet run

The preceding W4A16 arithmetic and GDN-normalization controls both return the
original wrong value. This next control removes only the extra BF16 cast of
normalized softmax routing weights relative to the GDN-normalization build.
The existing storage and expert accumulation already use FP32. Marlin consumes
FP32 routing weights; this still does not copy its expert-output staging.

The focused GPU check compares all three router launch forms at real geometry
(H=2560, E=512, K=10) and compares normalized weights to independent FP64 softmax.
It retains separate dot/selection checks and runs before the original request.
The real captured BF16 weights summed between 0.9981689453 and 1.0017089844;
436 of 576 sampled rows did not sum exactly to one. This is rounding, not yet
proof of a retrieval defect.

The original request runs on fresh TP2, then bounded prefill only if native
passes. Exact production restoration and inference verification run in finally.
