# Issue #4: repeated-reference boundary trace

Prepared, not executed. The pinned uninstrumented vLLM reference returned
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
