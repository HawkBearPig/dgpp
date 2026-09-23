# Issue #4: repeated-reference boundary trace

Executed on fresh TP2; peer capture collection is in progress. The pinned uninstrumented vLLM reference returned
5/6 and then 6/6 for two identical requests in one world. This temporary overlay
copies layer-boundary values to CPU without changing model arithmetic. It
captures full first-chunk arrays, last rows of later prefill chunks, and decode
rows through position 261319, plus exact per-call token IDs and positions.
Captured boundaries include embedding, PLE history arguments, attention input/
output, residual state, MoE input/output, injection coefficients and final hidden.

First compare the two generated answers with the uninstrumented 5/6 and 6/6
controls. Synchronization can affect a race; if the behavior changes, the trace
cannot be assumed representative. Otherwise find the earliest observed
boundary difference under identical token IDs, then narrow that operator with
frozen-input replay. Last-row traces after the first chunk do not prove that
an earlier unsampled row was identical.

`prepare_trace.py` generates the model overlay from the retained immutable image
source; `issue4_trace.py` is mounted as an additional read-only module. The
campaign restores production before collecting the peer's capture files.
This instrumentation is diagnostic and is excluded from the EOS correction.

The instrumented requests both scored 5/6, but returned different wrong values:
`val_5dac9ed720abddaf` (184 output tokens, 180.77 seconds), then
`val_5b9d2c3d7a95b986` (190 output tokens, 162.95 seconds). This does **not**
preserve the uninstrumented failure/success pair. Read-only synchronization
changes the observed result, so the trace is only a localization of repeated
execution variation under this instrumentation.

The first chunk's input tokens, PLE history, embedding, layer-0 attention input/
output, expert input, residual and injection fields are identical between the
two executions. The first observed difference is `layer0_mlp_out`: 4611 of
5242880 BF16 values across 150 rows, relative L2 8.63024e-5, maximum absolute
change 0.0009765625. There are no nonfinite values. All 340 first-chunk fields
match exactly **across ranks** separately for each request. This localizes
variation inside the first MoE block, not the preceding attention/state path or
unexpected disagreement at the captured replicated outputs. It does not yet
separate routing, shared expert, routed expert or final accumulation.

Variation grows through subsequent layers: first-chunk MoE-output relative L2
is about 0.099 by layer 3 and 0.197 by layer 47. These compare different input
trajectories, not operator errors. Metadata, differences and response verdicts
are retained in the top-level JSON files. Production restoration was verified
before peer collection. The separate expert-finalization control now tests
whether disabling the backend's fused accumulation removes the variation.

Both-rank comparison is complete (`rank-comparison.json`); the first difference
and counts agree on both ranks. Slow SFTP collection was replaced after verified
production restoration by a successful tar stream. The campaign's exit 1 is
confined to that intentionally interrupted copy step; `collection.json` records
successful recovery.
