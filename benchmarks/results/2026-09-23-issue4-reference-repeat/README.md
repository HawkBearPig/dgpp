# Issue #4: reference repeatability

A fresh TP2 reference world reuses the first run's vLLM/FlashInfer cache and
executes the unchanged original request twice, with prefix caching disabled.
The image, checkpoint, source overlay and engine arguments are identical to
the adjacent issue4-reference record. No arithmetic/source changes are made.
Both ranks load a cache byte-identical to the original rank-0 cache, SHA256
`f78838c7dee3ef4130abf3877af9192e1438fe7a42c7c8a3a087f4854db7ab36`.

| Request in fresh world | Checks | Failed-key value | Completion tokens | Seconds |
| --- | --- | --- | --- | --- |
| First | 5/6 | `val_5b9d2c3d7a95b986` | 190 | 124.27 |
| Second, unchanged | 6/6 | `val_aed39758a1abf065` | 187 | 122.96 |

The first response text and usage exactly match our prior fresh-world failure.
The second response text and usage exactly match the reporter's saved success.
Both requests stop normally; the original 261120 native token IDs match the
reference tokenizer. The metrics count two completed requests and 522240
prefilled tokens, with prefix caching disabled. The generic DGPP scorer marks
missing API cached-token details as a protocol failure; the top-level verdicts
account for the explicitly disabled reference prefix cache and preserve that
raw scorer result separately.

This is request-to-request variation under unchanged configuration, not yet a
localized kernel defect. Possible causes include execution nondeterminism or
state/allocation differences; it does not establish a DGPP defect. The successful
reference answer is now reproduced locally, but is not a stable expected-state
oracle. The proposed heuristic-tactic ablation is deferred: cache selection
alone cannot explain this within-world variation. A DGPP same-world repeat is
being tested to determine whether the analogous behavior occurs in the engine
under investigation.

Production was restored and independently verified after both requests.
