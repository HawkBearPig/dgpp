# Issue #4 exact-fixture follow-up — 2026-09-23

This investigation uses the reporter's original request, not the independent
seeded ledgers from the September 19 maintainer investigation. No production
model or kernel source is modified.

**Result: current master reproduces the exact reported misbinding.** Native
prefill and 1,024-token bounded prefill both return 5/6 correct associations,
with assistant content byte-identical to the reporter's clean-build response.

## Inputs and offline verification

The source is [the reporter's gist](https://gist.github.com/intobigdata/56c3957adb90ed09413d2d1da46f9d12).
Download each file from its pinned raw URL: the Gist API truncates the large
archive and can omit the other files' inline contents.

- Evidence archive SHA256:
  `674ea4ea000cf218c1d31ecb7c1b2816c94ebda17ca3fbd7ff9adbd8eb8fbe35`.
- Original request SHA256:
  `9ebc3a1ed20f37f3501bf4629bf84d3e0523cbb33989c3ce68fb5e8fd0db2393`.
- Oracle SHA256:
  `fb8dde5737c1a9bf78e89188f2fd8ba09bd7737faa51f6795458813d3ceafc64`.
- All 133 manifest entries verified by size and SHA256.
- The ledger has 7,328 unique keys, including the five named decoys. All five
  requested values and the absent-key expectation agree with the ledger.
- Independent rescoring of the archived responses gives DGPP 5/6 and vLLM
  6/6. These are saved-response checks, not new inference runs. vLLM's saved
  response has null `prompt_tokens_details`; the replay client's DGPP cache
  protocol gate is not applicable to that reference response.
- The native renderer linked against freshly built current-master libraries
  produced the same prompt bytes and every one of the 261,120 token IDs.
  Rendered-prompt SHA256:
  `532fa3ef6740047c572f6e6ab0a5a46891a069e3693f985a4fded8cec066f754`.
  The local tokenizer/template hashes match the reporter's retained metadata.

## Build and experiment

Clean source: `c6ca191863d620360f3b4370c0156f9e26a2631f`.
Release build: `cmake --preset release` then
`cmake --build --preset release -j4`, which builds `dgpp_serve_app`.
Version: `0.1.0+gc6ca191863d6`.
Executable SHA256:
`b6afe5546afb687e4f9adea276625b2cbff2ac38ba6dc60eab55879a50ddba47`.
Checkpoint: `nvidia/Qwen3.8-Flash-Next-NVFP4` at
`fc694b54fb0174e0913e6adf86691ef85a4ead47`.

Both cases use real TP2 over RoCE on the first two site Sparks. Each starts
a fresh process world and sends the original request as its first generation
request, without adding a grammar, changing message text, or changing the
1,024-token output reservation. Four slots, 262,144-token BF16 KV capacity,
FP8 dense weights, mmap n-gram table, MTP depth one and 1.5 GiB prefix cache
match the reporter's clean-run configuration.

`baseline.json` preserves the explicitly zero active and idle prefill
budgets in the reporter's resolved configuration. GEMV vocabulary head,
checkpoint BF16 storage, disabled compaction and absent RoPE scaling make
the original numerical choices explicit on current master. Current Qwen
includes its vision tower, but this request contains only text.

`bounded1024.json` changes only the active and idle prefill budgets to 1,024.
It measures the original free-generation request under another scheduling
shape. It does not reproduce the reporter's unpublished forced-prefix patch
or its internal trace comparison.

The running four-node GLM deployment was idle before the test window. Its
binary, resolved configuration, identities and metrics were saved before
shutdown. The campaign restores its original binary/configuration in a
`finally` block after stopping the temporary Qwen world.

## Executed results

| Case | Correct checks | Prompt / completion tokens | Cached tokens | HTTP / finish |
| --- | --- | --- | --- | --- |
| Native, budgets 0 / 0 | 5/6 | 261120 / 184 | 0 | 200 / stop |
| Bounded, budgets 1024 / 1024 | 5/6 | 261120 / 184 | 0 | 200 / stop |

Both return `key_0769e0226c63` → `val_5dac9ed720abddaf`; the correct value
is `val_aed39758a1abf065`. Assistant-content SHA256 in both cases and the
reporter's saved response:
`c2413f544aa63bf4294825a521fc229e79fd8c5eafaef1e57038850d92b5bbec`.

Before/after service counters establish that each was the first generation
request after its own server startup. Both rank configuration digests match
within each world, as do the operation-stream MD5s after clean shutdown.
Neither world's rank logs contain ERROR or FATAL lines. Binary identity was
also read directly from both running bounded-case processes and matched the
fresh build. See `results.json` and the two verdict files. Wall times are
recorded as receipts, not a controlled performance comparison.

This confirms that the merged changes through `c6ca191` do not resolve this
fixture. Changing these prefill budgets does not repair its visible answer.
Identical free-generation output does not establish internal numerical
identity and does not contradict the reporter's forced-prefix consistency
finding. Their patch and 261,290-token diagnostic fixture were not available
and were not tested. No fresh vLLM inference was run in this follow-up; its
6/6 result above is an independent rescore of the supplied response.

The original GLM service was restored at 05:41 UTC. Readback verified the
saved executable SHA256 on all four ranks, an identical resolved
configuration, HTTP health, and a successful `OK` inference with normal stop.
It was idle with no engine failure afterward. The smoke client initially
used Qwen's unsupported `enable_thinking` setting and received HTTP 400;
the successful GLM smoke uses `reasoning_effort: minimal`. Both receipts are
retained. No production source/binary update was deployed.

Review of the reporter's prefill-consistency patch is a separate track;
that patch is not required to investigate #4's semantic failure. The
[operator follow-up](../2026-09-23-issue4-operators/README.md) independently
reconstructed the exact forced prefix, matched its published token hash,
and captured attention/state boundaries on current master. Its numerical
checks proceed without waiting for the reporter.

## Replay

On reserved idle hardware, start the selected deployment with the fresh
binary, then run:

```sh
python3 replay.py --request raw/request.json --oracle raw/oracle.json \
  --out raw/native-run --url http://127.0.0.1:18084
```

The output directory must not already exist. The client checks model identity
and zero prior requests/prefills before sending the original bytes, saves
transport/response/metrics, rejects duplicate JSON keys, and checks all six
associations, normal stop, exact prompt count, zero cached tokens and exactly
one admitted request/prefill. A semantic failure exits 1 while preserving the
evidence.

Raw artifacts and machine-specific restoration material are ignored by Git.
The compact summaries and configuration files describe the experiment.

## Reporter follow-up

[Requested the unpublished patch and missing test artifacts](https://github.com/HawkBearPig/dgpp/issues/4#issuecomment-5789566019):
exact base/commit or diff, focused regressions and bounded/native controls,
the 261,290-token forced-prefix fixture/replay commands, and any original-request
result on the patched executable. The original reproduction information is
already complete.

The follow-up reconstructed the forced-prefix fixture, so the reporter request
was updated to remove that request and clarify that patch review is independent
of continued root-cause investigation.
