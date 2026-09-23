# Reference expert-router tie handling — complete

264 of 576 real captured logit rows have exact BF16 ties at the tenth-expert
selection boundary. DGPP uses lower expert IDs on ties. The independent
numeric MoE audit validated DGPP's policy, but did not compare reference tie
choices on identical logits.

This check calls the pinned reference's actual `fused_topk` router on the
captured logits, at both original prefill batch sizes and one row at a time.
It compares selected expert sets, reports whether any swaps are confined to
exact ties, and checks normalization/rounded weights and call-shape agreement.
No model accuracy conclusion follows from this operator check alone.

All 576 rows choose identical expert sets at both prefill batch shapes and
for individual rows. IDs and FP32 weights repeat exactly across call shapes;
weights rounded to BF16 exactly match DGPP captures. Maximum weight-sum error
is 2.384185791015625e-7. The test container was removed before the TP2 model
world started. This rules out differing tie handling on these identical
logits, not differences in logits produced earlier in the model.
