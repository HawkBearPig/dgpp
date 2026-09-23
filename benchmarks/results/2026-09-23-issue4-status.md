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
| [FP32 routing weights](2026-09-23-issue4-router-fp32/README.md) | Building | Removes the extra BF16 cast after top-k renormalization |

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
The next control retains normalized expert routing weights in FP32.
