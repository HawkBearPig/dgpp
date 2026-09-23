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
