# Issue #4 investigation status

The original near-262K retrieval failure remains reproducible and unlocalized.
No existing PR or tested candidate resolves it. The unpublished prefill
consistency patch is not a prerequisite for the investigation.

| Experiment | Executed result | What it establishes |
| --- | --- | --- |
| [Clean native and bounded replay](2026-09-23-issue4-exact/README.md) | Both 5/6, original wrong value, 261120 prompt tokens | Failure persists on current master with the unchanged request |
| [Forced-prefix operator replay](2026-09-23-issue4-operators/README.md) | Captured and clean predictions/logprobs match; independent attention/projection checks stay within numerical contracts | No localized mismatch at the tested boundaries; captured inputs can already be wrong |
| [Full-history state/cache audit](2026-09-23-issue4-history/README.md) | 49664 exact continuity/cache checks; sampled-head recurrence and full PLE history replays pass | No detected state handoff/cache corruption; not a complete independent model execution |
| [Trained EOS preservation](2026-09-23-issue4-eos/README.md) | Real configuration defect fixed; original request still 5/6 | Separate confirmed bug, submitted as draft PR #34 |
| [NVFP4 activation quantization](2026-09-23-issue4-precision/README.md) | Quantizer agrees with independent oracle; request still 5/6 with a different wrong value | W4A4 emulation alone does not fix the failure |
| [Combined EOS, W4A4 and checkpoint dense](2026-09-23-issue4-trained/README.md) | Original request still 5/6 | This combination is insufficient; not exact reference arithmetic |
| [Gated-residual FP32 intermediates](2026-09-23-issue4-gr-precision/README.md) | Candidate follows reference rounding policy more closely; request still 5/6 | Keep this arithmetic change diagnostic |
| [Fresh pinned vLLM reference](2026-09-23-issue4-reference/README.md) | All 261120 token IDs match; same key fails, 5/6 | A single reference success is not a stable accuracy oracle |
| [Answer-order controls](2026-09-23-issue4-answer-order/README.md) | Forced first answer selects adjacent decoy; reordered query selects all five decoys | Strong order sensitivity; not proof of an engine defect |
| [Reference repeatability](2026-09-23-issue4-reference-repeat/README.md) | First request 5/6, identical second request 6/6 | Same-world variation despite identical cached tactics; saved success reproduced |
| [DGPP repeatability](2026-09-23-issue4-dgpp-repeat/README.md) | Both 5/6, identical text/usage and zero cached tokens | No analogous variation in the tested DGPP world |
| [Reference boundary trace](2026-09-23-issue4-reference-trace/README.md) | First full-chunk difference at layer-0 MoE output under identical inputs | Reference itself introduces numerical variation |
| [Reference non-fused finalization](2026-09-23-issue4-reference-finalize/README.md) | All full first-chunk fields repeat exactly; later QSA output still varies | One reference variation source isolated; original failure remains |
| [Deterministic reference QSA](2026-09-23-issue4-reference-deterministic/README.md) | Both 5/6, same wrong answer; all 108120 captured fields per rank identical | A repeatable reference still fails the key |
| [Post-dot activation scales and reference rounding](2026-09-23-issue4-postscale/README.md) | Invalid control: stale resident image bypassed changed loader globals | Quantizer passes; integration must be rerun without weight-image cache |
| [FP8 checkpoint control](2026-09-23-issue4-fp8-control/README.md) | 5/6, different wrong value, 174 output tokens; input and all checked non-expert language tensors match | Higher expert weight precision alone does not repair the failure |
| [Checkpoint rebuild](2026-09-23-issue4-uncached/README.md) | Same 5/6 answer and usage, with resident images bypassed on both ranks | Cached weights do not explain the observed failure |
| [Uncached arithmetic control](2026-09-23-issue4-postscale-uncached/README.md) | 5/6; full answer and token counts match deterministic reference exactly | Correct scale integration changes the wrong answer, but does not repair retrieval |
| [W4A16 reference](2026-09-23-issue4-reference-w4a16/README.md) | Two identical 6/6 answers, 187 completion tokens | Repeatable successful reference in the tested world; other arithmetic differences remain |
| [W4A16 DGPP rounding control](2026-09-23-issue4-reference-rounding-w4a16/README.md) | 5/6, same original wrong answer and 184 output tokens | GR/SwiGLU reference rounding with checkpoint dense projections is insufficient |
| [Real-fixture expert replay](2026-09-23-issue4-moe-audit/README.md) | Exact capture parity; 576 local and 288 folded chains pass | No detected routed/shared expert error on sampled inputs |
| [GDN normalization policy](2026-09-23-issue4-gdn-norm/README.md) | Focused GPU tests pass; same original 5/6 answer | Lower operator error is insufficient to repair retrieval |
| [FP32 routing weights](2026-09-23-issue4-router-fp32/README.md) | GPU router oracle passes; original request still 5/6 | Extra BF16 route-weight rounding is insufficient to explain the failure |
| [Untraced deterministic W4A4 reference](2026-09-23-issue4-reference-w4a4-untraced/README.md) | Invalid; aborted before verdict | Explicit backend option changed the cache identity and retuned 17 tactics |
| [GDN normalized Q/K boundary](2026-09-23-issue4-gdn-qk-staging/README.md) | Four GPU tests pass; unchanged original answer remains 5/6 | Normalized-Q/K BF16 rounding is insufficient to repair the failure |
| [Expert output storage](2026-09-23-issue4-moe-staging/README.md) | Focused GPU check passes; same original 5/6 answer | Weighted-expert/shared BF16 stores are insufficient |
| [Main Q/K rotation storage](2026-09-23-issue4-qsa-rope-staging/README.md) | Nine GPU tests pass; original request still 5/6 | FP32 main Q/K rotary products are insufficient to repair retrieval |
| [Frozen GDN comparison](2026-09-23-issue4-gdn-frozen-reference/README.md) | 16 cases, all finite; recurrent capture agreement ≤0.033% output L2, chunk/recurrent difference ≤0.371% | Isolates normalized-input storage from recurrent/chunked arithmetic |
| [Untraced W4A4 retry](2026-09-23-issue4-reference-w4a4-untraced-retry/README.md) | Two identical 5/6 answers; full text/usage match traced reference | Tracing does not explain observed W4A4/W4A16 difference |
| [W4A16 recurrent-GDN reference](2026-09-23-issue4-reference-recurrent-gdn/README.md) | Two identical 5/6 answers, exact DGPP text/token counts | A2 also fails: GDN causal attribution is not established |
| [Fresh W4A16 repeat](2026-09-23-issue4-reference-w4a16-repeat/README.md) | Two original 5/6 answers with unchanged GDN | Earlier W4A16 success is not stable across fresh worlds |
| [Expert router ties](2026-09-23-issue4-router-ties/README.md) | All 576 captured rows agree across prefill/decode batch shapes | No reference/DGPP expert-set differences on identical logits |
| [Canonical reference grouping](2026-09-23-issue4-reference-canonical-moe/README.md) | Two original 5/6 failures; all captured fields repeat exactly | Removing measured reference grouping variation does not repair retrieval |
| [Reference FP32 state](2026-09-23-issue4-reference-state-fp32/README.md) | Two original 5/6 failures; actual FP32 handoffs verified | BF16 reference state storage is insufficient to explain the failure |
| [Independent FP8 reference](2026-09-23-issue4-reference-fp8/README.md) | Two 5/6 failures with the original wrong association, 169 output tokens; 257280 fields repeat exactly | Higher expert precision in the independent engine is insufficient |
| [FP8 experts + BF16 PLE table](2026-09-23-issue4-fp8-ple-precision/README.md) | A/B/A all 5/6; table precision changes the wrong value; baselines repeat exactly | Combined higher expert/table precision is insufficient |
| [Full PLE projection/gate and QSA keys](2026-09-23-issue4-ple-projection/README.md) | All 261290 PLE rows; 24 QSA key chains; exact frozen GPU reproduction of 29 outliers | Local key-projection differences are amplified by the gate; retrieval causality remains untested |

[Draft PR #34](https://github.com/HawkBearPig/dgpp/pull/34) separates generation
stop IDs from trained model EOS, preserving PLE semantics. The release build,
22/22 host CTest entries and focused EOS regression passed. The two TP2
operation-stream files have identical MD5 `de47344a6aeadd284e4cdf230408cbbe`.
The original request still fails, and the PR explicitly does not close #4.

The reference's immutable container, source overlay, checkpoint, tokenizer
IDs, library versions and effective engine configuration match the supplied
recipe. Both local checkpoint snapshots passed complete content-hash checks
(25 files, 132734506847 bytes per node). The saved successful reference's
FlashInfer autotuning cache and fresh-world repeatability were requested in
[this issue comment](https://github.com/HawkBearPig/dgpp/issues/4#issuecomment-5790840855).
Source tracing confirms that cache selects tactics on the active CUTLASS MoE
path. Its causal role is not established.

The reference varies across identical requests in one world. Its second answer
exactly reproduces the saved 6/6 success, whereas its first reproduces our earlier
5/6 failure. DGPP repeats its original wrong answer exactly in two same-world requests.
The traced reference first varies at layer-0 MoE output; disabling fused
finalization removes all first-chunk variation. The remaining first sampled
difference is QSA layer 3 in the second chunk. The deterministic reference top-k
control repeats the same 5/6 answer with identical captured fields on both ranks.
DGPP already selects score/index ties deterministically.
Do not change production arithmetic solely to fit one sampled reference answer.

GPU campaigns use reserved idle hardware serially and restore the exact
original four-node production binary/configuration with a verified inference
smoke test. Raw captures and receipts remain local under each record's ignored
`raw/` directory. The EOS fix is on branch `fix/qwen-model-eos`; separate diagnostic branches
use `/tmp/dgpp-issue4-20260923`. The user's original tracked source is unchanged.

The [real-fixture MoE audit](2026-09-23-issue4-moe-audit/README.md) passed all
576 local and 288 folded chains, including routing/segmentation invariants. Its
first, incorrectly configured attempt was quarantined; the valid capture has
exact prediction/logprob/usage parity with the clean forced-prefix control.
GDN normalization with FP32 intermediates has lower independent FP64 error in
all 72 sampled rank/layer cases, but still returns the original 5/6 answer.
FP32 routing weights, normalized GDN Q/K storage, expert output storage and
main Q/K rotation storage all leave the original 5/6 answer unchanged.
Their focused GPU checks pass, but no retrieval fix is established.

The frozen reference GDN comparison completes all 16 first/final-chunk cases.
Its recurrent FP32-normalization path closely matches the original captured
DGPP recurrence; holding normalized inputs fixed separates the reference's
chunked arithmetic difference. The subsequent GDN swap preserves BF16-normalized inputs, gates and decode.
The fresh unmodified repeat also fails, preventing causal attribution.

The untraced W4A4 retry completed two identical 5/6 answers, matching the
traced deterministic reference's full text and usage. Both runtime-selected
cache identities/hashes match the retained cache and stayed unchanged after
inference. Its QSA/helper/PLE overlays match the successful Marlin world.
The earlier explicit-backend attempt is invalid because it changed cache
identity and retuned 17 tactics. Two retry startups failed harness checks
before inference and were quarantined with verified production restoration.

Reference expert routing also agrees with DGPP on all 576 captured logit rows,
including 264 exact ties at the tenth-expert boundary, across both prefill
shapes and individual rows. This does not test earlier logit divergence.
The recurrent GDN source adapter passed real-geometry GPU checks. Both full
requests returned the original DGPP failure with exactly matching full text
and token counts (184 completion tokens). Production restoration passed.

A fresh [unmodified W4A16 repeat](2026-09-23-issue4-reference-w4a16-repeat/README.md)
returned the original DGPP failure twice, with exact full text/token-count
agreement. Original GDN source hashes and Marlin selection were verified on
both ranks. This invalidates causal attribution from the GDN swap.

The [native chunked GDN prototype](2026-09-23-issue4-gdn-chunk/README.md)
passes 24 operator comparisons and five executed GPU tests, but the unchanged
original request still scores 5/6 with a different wrong record value,
`val_7265273163533973` (190 completion tokens). Production restoration passed.
The prototype is diagnostic and is not a proposed retrieval fix. A separate sixth
fixture test subsequently passed on GPU.

A [fresh read-only Marlin trace](2026-09-23-issue4-reference-world-trace/README.md)
reproduces the original 5/6 answer and exactly matches the untraced failure's
text/usage. All 267480 captured fields across both ranks pass payload-hash
verification. It confirms BF16 state storage between reference calls despite
FP32 within-call accumulation; DGPP and the checkpoint require FP32 state.
This is an additional numerical-policy difference, not a localized retrieval
cause. The second fresh trace also fails 5/6, with identical full text/usage, but its
first layer-0 MoE output differs on both ranks in 14/5242880 elements under
identical captured preceding inputs and identical recorded GDN configurations.
The detailed [cross-world comparison](2026-09-23-issue4-reference-world-trace-repeat/README.md)
accounts explicitly for an incomplete final trace call. This localizes reference
variation to MoE, not yet to a specific internal operation or the retrieval cause.
An explicit FP32-state reference control and a bounded internal-MoE probe are
prepared. The [native block-GDN/checkpoint-dense control](2026-09-23-issue4-gdn-checkpoint-dense/README.md)
still returns the original 5/6 answer (184 completion tokens). Both rank loader
receipts confirm checkpoint dense weights and no resident image loads. The
bounded follow-up was skipped and production restoration passed. An earlier
pre-inference log-collection guard failure was corrected and is excluded from
accuracy results.

The [bounded Marlin probe](2026-09-23-issue4-reference-marlin-probe/README.md)
now causally localizes reference expert variation to token grouping order. Both
rank inputs exactly match the original first chunk; routing/shared outputs remain
exact while valid token placement and the first routed GEMM vary. In four fresh
processes, automatic grouping produces ten different outputs per ten replays;
all 40 frozen-grouping replays exactly reproduce every captured expert stage.
DGPP already uses deterministic placement. This is a reference reproducibility
finding, not a DGPP retrieval fix.

The [canonical grouping control](2026-09-23-issue4-reference-canonical-moe/README.md)
passes 60 GPU/CPU-oracle cases and repeats exactly in 40 real-operand replays
across fresh processes and both ranks. The full TP2 campaign completed two
unchanged original requests with canonical grouping, deterministic QSA and
pinned GDN launch choices, retaining BF16 auto state. Both return the original
5/6 answer with identical full text/usage (184 completion tokens). All 267480
captured fields across both ranks repeat exactly, with no missing final fields;
all 534960 saved payloads pass shape, length and SHA256 checks. Production
restoration passed.

The [FP32-state comparison](2026-09-23-issue4-reference-state-fp32/README.md)
also returns the original 5/6 answer twice, with identical text/usage. All 267480
captured fields repeat exactly and all 534960 payloads pass integrity checks.
Both ranks match the BF16-state control by value throughout the first chunk;
the first numerical difference is the layer-0 state read at position 2048.
All 12 checked handoffs preserve FP32 state exactly. The intended state-storage
change affects later computation but does not repair retrieval. Production
restoration passed.

The [independent FP8 reference](2026-09-23-issue4-reference-fp8/README.md)
now completes the original input twice: 5/6, original wrong association,
169 completion tokens, identical full text and usage. All 48 sampled actual
loaded weight/scale checks pass. Both ranks match the NVFP4 reference in all
17 captured fields before the first expert output; that output is the first
numerical difference. All 257280 captured fields across ranks repeat exactly,
all 514560 saved payloads pass integrity checks, and all 12 checked state handoffs
preserve FP32 bits. Production restoration and peer capture collection passed.
Higher expert precision with this independent backend does not repair retrieval.
An earlier diagnostic assertion about checkpoint scale dtype was corrected and
that aborted attempt is explicitly excluded from accuracy evidence.

A full BF16 checkpoint is available, with identical text configuration to FP8.
It is an optional broader quantization control, not a prerequisite for a DGPP
fix; no full BF16 model run has started. The narrower
[PLE precision comparison](2026-09-23-issue4-ple-quantization/README.md) is now
complete. Small BF16 table samples show no large conversion/scaling
discrepancy, but have a measured 2.67% relative FP8 quantization difference.
The diagnostic replaces only n-gram table values while retaining native
NVFP4 TP2 execution. Its server builds and all 11 focused GPU/loader checks pass,
including an eager/MTP-graph storage control. All selected table files and the
capture relocation pass full checksum verification. The original-request A/B/A
comparison returns the exact original failure in all three runs: 5/6, 184
completion tokens, identical full text and usage. Both current-launch loader
logs and actual process environments confirm the precision change only in the
middle run. Effective configuration and actual binaries match, and the campaign
exits 0 after verified production restoration. Higher table precision alone
does not repair the native NVFP4 failure. An earlier failed
diagnostic copy path was corrected before any original request ran and is
excluded from accuracy evidence; that attempt restored production successfully.

The missing [FP8-expert/BF16-table combination](2026-09-23-issue4-fp8-ple-precision/README.md)
is now complete using cached weights and the same validated diagnostic binary.
A1/A2 exactly reproduce the earlier native FP8 failure (`val_51bf3e137245e273`,
174 output tokens); B returns another existing record's value
(`val_7265273163533973`, 175 output tokens). All score 5/6. The original request
changes only the served model name and retains all 261120 token IDs. Actual
process binary/environment checks and current-launch loader logs verify the
change only in B; effective engine configuration matches. Resident images are
disabled throughout, and A1 text/usage exactly match the prior cache-enabled
control. Both campaign and independent rescorer exit 0; production restoration
passes all four binary checks, exact configuration, smoke inference and idle
state. All four tested expert/table precision combinations fail the same key.
This rules out those precision changes as sufficient repairs, not every
quantization effect or every possible engine defect. No full BF16 model was
run. The retrieval root cause and a causally supported fix remain unresolved.

The [complete PLE projection/gate audit](2026-09-23-issue4-ple-projection/README.md)
closes the earlier checkpoint-to-gate coverage gap across all 261290 saved rows.
All eight source hashes and replicated rank fields match. Aggregate gate error
against independent FP64 projection arithmetic is 0.0257%, but the worst row
has 1.8765% relative error. All 32.8 million PLE projection weight conversions
match the production encoder bit-for-bit. A bounded GPU replay at native row
placement exactly reproduces all gate and normalized outputs for the 29 worst
rows. The gate itself agrees bit-for-bit when given actual GPU intermediates;
isolated key normalization cannot explain the outliers. The dominant difference
comes from projected keys and their nonlinear amplification. An FP32-output
control with the original BF16 partial boundaries returns identical outputs.
The separate final-chunk QSA key-projection/norm/RoPE audit covers 7.34 million
values, with worst per-layer/rank L2 0.0242% and exact cached-tail copies.
Production was restored and verified after the bounded GPU probe. These are
operator findings, not proof of the retrieval cause. They motivated the
isolated higher-accuracy key-projection control below.

The [FP64 PLE key-projection control](2026-09-23-issue4-ple-fp64-key/README.md)
is now complete. The frozen GPU replay exactly matches the independent key
projection/fold oracle at both native prefill shapes, corrects the identified
gate outliers as predicted, and preserves the other operand paths. Nevertheless,
the byte-identical original request returns the original 5/6 failure in all
three fresh native/FP64/native TP2 worlds, with identical full text and usage
(184 output tokens). Actual binaries, process environments and execution logs
verify the change on both ranks only in the middle run. Production restoration
passes all four original binary hashes, exact configuration, inference and idle
state. The numerical lead is insufficient to repair retrieval; the experimental
patch remains a diagnostic, not a proposed fix. No full BF16 model was run.

A [final-row residual-boundary replay](2026-09-23-issue4-gr-boundaries/README.md)
also closes a narrower coverage gap. Independent checkpoint arithmetic carries
both residual injections through all 48 layers, accepting captured preceding
residuals and folded attention/MoE branch outputs. All 491520 final residual
values agree bit-for-bit; the intervening MoE input mixer has worst relative L2
0.000966905. Both ranks' fields agree exactly. The PLE layer is now included
using a bounded export from freshly rehashed complete post-PLE files. This
checks the failed row's residual boundaries, not the whole prompt's earlier
construction. The retrieval root cause and a causally supported fix remain
unresolved; no additional reporter artifact is required for further local work.
