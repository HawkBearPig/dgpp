# Expert output storage boundaries — original request still fails

Relative to the GDN normalized-Q/K control, this diagnostic adds the pinned
Marlin reference's BF16 storage boundaries: each weighted expert down result,
then the FP32 sum to BF16; shared down to BF16, weighted shared result to BF16,
and final BF16 addition. It keeps the existing deterministic expert order.
It does not reproduce Marlin's GEMM reduction order or the entire FLA recurrence.

The diagnostic changes both segmented and decode-slot routed accumulators,
and both ordinary and fused shared tails. It currently changes the generic
FP32 routed interfaces used by MiMo too; this is not a production-ready patch.
A positive retrieval result requires a minimal, Qwen-scoped ablation and broader
validation before promotion. Existing model oracles still encode the original
policy and are not asserted to pass this diagnostic.

A focused independent FP64 staging check covers segmented/slot accumulation
and shared combination, with a fixture required to distinguish the old policy.
The unchanged original TP2 request follows only after that check; bounded
prefill follows only if native succeeds. Production restoration runs in finally.

The GPU storage test passes all 768 outputs exactly, with 295 outputs that
distinguish the old policy. On frozen captured inputs the reference storage
policy changes 563920/1474560 routed BF16 outputs across 576 cases. The original
request nevertheless returns the same 5/6 answer and 184 completion tokens.
Bounded prefill was skipped. Production restoration and independent identity,
configuration and inference checks passed.
