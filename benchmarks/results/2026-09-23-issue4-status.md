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
| [Fresh pinned vLLM reference](2026-09-23-issue4-reference/README.md) | All 261120 token IDs match; same key fails, 5/6 | The saved reference success is not yet reproduced locally |
| [Answer-order controls](2026-09-23-issue4-answer-order/README.md) | Forced first answer selects adjacent decoy; reordered query selects all five decoys | Strong order sensitivity; not proof of an engine defect |
| [Reference repeatability](2026-09-23-issue4-reference-repeat/README.md) | First request 5/6, identical second request 6/6 | Same-world variation despite identical cached tactics; saved success reproduced |
| [DGPP repeatability](2026-09-23-issue4-dgpp-repeat/README.md) | Both 5/6, identical text/usage and zero cached tokens | No analogous variation in the tested DGPP world |
| [Reference boundary trace](2026-09-23-issue4-reference-trace/README.md) | Prepared | Locate repeated-reference divergence before changing DGPP arithmetic |

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
5/6 failure. The next test checks DGPP same-world repeatability; further reference
localization must distinguish execution nondeterminism from state/allocation
differences before using a saved answer as an expected hidden-state oracle. Do not change production arithmetic solely to fit this single answer.

GPU campaigns use reserved idle hardware serially and restore the exact
original four-node production binary/configuration with a verified inference
smoke test. Raw captures and receipts remain local under each record's ignored
`raw/` directory. The EOS code is isolated in `/tmp/dgpp-issue4-20260923`, branch
`fix/qwen-model-eos`; the user's original source checkout is unchanged.
