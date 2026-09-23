# PLE key projection FP64 accumulation — validated control, retrieval still fails

Diagnostic implementation and original-request A/B/A, following the
[frozen projection/gate audit](../2026-09-23-issue4-ple-projection/README.md).
No retrieval fix is established by implementing or passing the operator control.

`DGPP_ISSUE4_PLE_FP64_KEY=1` changes only PLE key products above 128 rows.
The original FP8 projection weights are dequantized through the existing BF16
bridge; operands are widened exactly and multiplied with cuBLAS DGEMM.
Each result is rounded FP64 → FP32 → BF16 before the unchanged rank fold,
normalization and gate. Value projections, query normalization, lookup storage,
small-row decode, MTP and the checkpoint are unchanged. The helper reconverts
weights on every call, including after resident views are rebound.

The isolated worktree starts at d318815 (the separate EOS configuration fix).
`diagnostic.patch` is experimental, not a proposed production fix. The default
path is unchanged. Additional scratch is accounted for in PLE memory planning.

Before any long request, the bounded replay must reproduce all previously
captured native fields exactly and match the independent FP64 key-projection
oracle bitwise at both 2048- and 1024-row shapes. Value and query fields must
remain unchanged. The same frozen operands and original 2048-row offsets are
used; the 1024-row check uses the first half of that input and tests the smaller
call shape, not a capture of the actual final chunk.

The serial campaign runs the byte-identical original NVFP4 request in three
fresh TP2 worlds: native, FP64 key accumulation, native. It verifies the
executable and environment on both ranks, logs actual FP64 prefill execution,
compares resolved configurations, and checks exact baseline response/usage
repeatability. Its `finally` path restores and verifies the original production
executables, configuration, inference and idle state.

## Executed results

The release server and bounded probe build successfully. The probe's native
mode reproduces all ten previous intermediate/output fields bitwise, including
all original outlier gate and convolution-normalization values. The FP64 mode
matches the independent key-partial/fold oracle on all 1320960 checked scalars
across the two shapes. Zero-padded rows remain zero; value and query fields are
unchanged. The new gate output exactly matches the independent key-only hybrid
intervention. At the worst position 136732, its first gate is 0.0306396484375,
as predicted, rather than the captured native 0.03173828125.

| Fresh TP2 run | Correct | Failed key's value | Completion tokens | Seconds |
| --- | ---: | --- | ---: | ---: |
| Native A1 | 5/6 | `val_5dac9ed720abddaf` | 184 | 218.911 |
| FP64 key B | 5/6 | `val_5dac9ed720abddaf` | 184 | 235.505 |
| Native A2 | 5/6 | `val_5dac9ed720abddaf` | 184 | 218.393 |

All three assistant texts and complete usage objects are identical. Each
request is byte-identical to the original (SHA256
`9ebc3a1ed20f37f3501bf4629bf84d3e0523cbb33989c3ce68fb5e8fd0db2393`),
contains 261120 prompt tokens and is the fresh world's first generation.
Actual process hashes/environments and current-startup logs verify execution
on both ranks only in B. Resolved configurations match after excluding log
directory paths. The server SHA256 is
`948ae69ec973bb137d064251db4053c77759df73d0b9e8c8330475a9b2f25294`.

The campaign and independent rescorer exit successfully. Production restoration
passes all four original binary checks, exact resolved configuration, an `OK`
inference and idle/nonfailed state. Raw operands, outputs, logs and serving
receipts are retained locally; compact validated results are in
`gpu-validation.json`, `comparison.json` and `restoration.json`.

This isolates and corrects the observed PLE projection numerical differences
but does **not** repair the original retrieval failure. No production arithmetic
change or retrieval-fix PR is warranted from this negative control. It does not
prove that every other PLE computation or the full earlier model history is
correct. No full BF16 checkpoint run was needed or performed.
