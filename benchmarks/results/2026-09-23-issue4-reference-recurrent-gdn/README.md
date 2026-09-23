# Successful W4A16 reference with recurrent GDN prefill — prepared, not run

Relative to the successful two-request untraced Marlin W4A16 reference, the
only model-source change replaces the chunked GDN prefill call with the
reference's fused recurrent operator. It retains the same prepared normalized
BF16 Q/K, V, FP32 gates, initial state and output layout. Decode is unchanged.
The single-sequence experiment supplies the serving recurrent kernel's valid
state-index slots explicitly and preserves caller input state by copying it.

This tests whether the chunked recurrence's arithmetic explains the remaining
DGPP/reference difference. It does not presume that either implementation is
incorrect. It follows the frozen-input kernel comparison and the untraced
W4A4 control. Both workers must confirm the overlay and Marlin selection.
Two unchanged original requests and verified production restoration are planned.

Before loading the model, a focused check runs the actual source adapter on
replicated frozen inputs with the real TP2 grouped-value geometry (8 key
heads, 24 value heads). It checks state preservation, optional output-buffer
contents, exact direct-operator agreement and repeatability. This is an API
check, not a retrieval result or full performance benchmark.

Source review identifies explicit chunked arithmetic boundaries: the triangular
inverse is stored in the key dtype (BF16); weighted K/V, reconstructed W/U and
corrected values also have BF16 boundaries; state views are cast to BF16 for
matrix products while persistent state remains FP32. The sequential recurrence
carries the state update in FP32. `fla-source.json` pins the inspected files.
Neither numerical policy is labeled incorrect on this evidence alone.

A failure after this swap would be a lead, not a complete causal verdict: the
unmodified Marlin baseline must then repeat successfully in another fresh
world (A/B/A) before attributing the change to GDN arithmetic. The existing
6/6 baseline consists of two requests in one world.
