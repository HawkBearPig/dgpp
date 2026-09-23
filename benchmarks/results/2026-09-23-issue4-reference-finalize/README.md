# Issue #4: reference expert-finalization control

Prepared, not executed. This one-line reference overlay disables fused MoE
finalization, which otherwise permits BF16 atomic accumulation. It is contingent
on the current repeated-reference trace implicating expert output. It is not a
DGPP fix or evidence that nondeterministic finalization caused the saved result.

A future run must use a separate FlashInfer cache root, retaining only the
original GEMM1 entries and allowing GEMM2 to retune: disabling finalization
changes the list of eligible GEMM2 tactics, so original GEMM2 numeric IDs must
not be reused. Compare repeated original requests and unchanged-input expert
boundaries before drawing conclusions. Keep the original cache untouched.
