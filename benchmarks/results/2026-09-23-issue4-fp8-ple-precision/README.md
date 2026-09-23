# FP8 experts with BF16 n-gram tables — completed, still 5/6

Replacing the FP8 PLE table with its original BF16 values still scores 5/6.
The failing key remains `key_0769e0226c63`: baseline FP8 tables return
`val_51bf3e137245e273`; BF16 tables return `val_7265273163533973`.
The expected value is `val_aed39758a1abf065`. Both wrong values belong to
other records in the original input; `value-origins.json` retains the lines.
The other five checks pass. The final baseline exactly reproduces the first
baseline's text and usage. Production was restored at 17:48 UTC; all four
original binary identities, the exact resolved configuration, an `OK` inference
smoke check and idle/nonfailed service state were independently verified.

| Run | Table values | Correct | Wrong value | Output tokens | Wall seconds |
| --- | --- | --- | --- | --- | --- |
| A1 | FP8 | 5/6 | `val_51bf3e137245e273` | 174 | 230.488 |
| B | BF16 | 5/6 | `val_7265273163533973` | 175 | 231.034 |
| A2 | FP8 | 5/6 | `val_51bf3e137245e273` | 174 | 230.679 |

The campaign and independent rescorer both exited 0. Accuracy failure is an
expected diagnostic result and is explicitly retained in every verdict.

## Controlled comparison

All runs use fresh native TP2 worlds and the existing tested diagnostic binary
`ea1a66ed60278d2d4f08d8fc7374c78cbd036fc07c6d33460b69ce11192fee8c`,
based on EOS-corrected commit `d318815984fd2ab1f0a67a6b830b57730a57439a`.
The expert checkpoint is `Qwen/Qwen3.8-Flash-Next-FP8` at
`236dfdf285828023ca3bcd3f37366c58a3469b13`. The BF16 table comes from
`Qwen/Qwen3.8-Flash-Next` at `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
Only the middle run enables `DGPP_ISSUE4_PLE_BF16_MANIFEST`.
The same binary runs on both ranks, and the actual process environments
are retained. Loader messages verify the BF16 row ranges and scale bypass.

The original request changes only its served model name. The earlier native
text/template check confirms all 261120 IDs match; `input-parity.json` binds
that receipt and the original and FP8 request hashes. The original oracle and
scoring code are reused. No shorter prompt or changed generation settings are
substituted. Requests are the first generation in each fresh world, with zero
cached tokens. All three runs use original engine settings: BF16 KV, FP8 dense,
MTP depth one, four slots, 262144 capacity and native prefill.

Resident images are disabled for all runs to avoid creating another incomplete
51 GiB cache on the head. This changes loading storage, not weight conversion.
The first baseline exactly reproduces the prior native FP8 result's complete
text and usage (`prior-baseline-parity.json`). The diagnostic passed 11 focused
GPU/loader checks in the preceding NVFP4 experiment; this campaign verifies
the unchanged test executables, server, source patch and test log hashes.
No full BF16 model was run and no additional weights were downloaded.

## Interpretation

The table change affects generation with FP8 experts but does not repair the
association. Together with the prior NVFP4 comparison, this closes the missing
combination in the tested expert/table precision matrix. It does not establish
that every operator is correct or that all quantization effects are irrelevant.
The experiment retains quantized experts and FP8 dense projections, and is
not a full BF16 model reference. No retrieval fix is supported by this result.

`campaign.py` owns serial cluster startup, replay and restoration in `finally`.
`summarize.py` independently rescores all three responses and checks input,
configuration, runtime binary, table override, current-launch loader receipts
and production restoration. Large inputs and raw logs remain under this
record's ignored `raw/campaign` directory in the primary workspace.
