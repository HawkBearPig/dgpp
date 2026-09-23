# Independent FP8 reference — original association still fails twice

The pinned reference ran the unchanged original input twice using
`Qwen/Qwen3.8-Flash-Next-FP8@236dfdf285828023ca3bcd3f37366c58a3469b13`.
Only the served model name changed; every one of the 261120 input IDs matches.
Both responses score **5/6**, returning `val_5dac9ed720abddaf` for
`key_0769e0226c63` instead of `val_aed39758a1abf065`. Both stop normally with
169 completion tokens. Wall times are 386.211 and 369.186 seconds.

Full assistant text and usage repeat exactly. The parsed answers equal the
NVFP4 reference's original failure, while JSON formatting and completion counts
differ. The earlier native FP8 control also failed this key, with a different
wrong record value. Higher expert precision in this independent reference does
not repair the fixture; no DGPP retrieval fix is established.

## Controls and validation

The reference uses its stock Triton W8A8 path. Its loader losslessly refines
128-by-128 weight scale blocks to 64-by-64 for TP2's 320-wide expert partition.
Actual loaded raw FP8 weights and refined FP32 scales pass all 48 independent
checkpoint-slice checks across four sampled experts on each rank. The checkpoint
scales are BF16 and are promoted exactly to FP32. This is sampled loader
validation, not a complete independent execution of every expert.

Explicit FP32 recurrent state, deterministic QSA, canonical expert grouping and
pinned GDN launch choices are retained. All 12 checked state handoffs preserve
FP32 bits exactly. Both ranks repeat all 128640 captured fields, with no input
or token drift and no incomplete tail (257280 equal fields across ranks).
All 514560 retained payloads pass shape, byte length and SHA256 checks:
6581886928 bytes, 297 calls per request/rank, and no orphan payloads.

On both ranks, all 17 captured fields preceding the first expert output match
the NVFP4 FP32-state control exactly. The first difference is the layer-0 expert
output: relative L2 0.0301552, maximum absolute difference 0.03125, with no
nonfinite values. Expert weights and backend activation arithmetic change
together; this does not isolate one NVFP4 arithmetic operation.

CPU-only inspection of the pinned configuration class confirms equal effective
routing normalization, PLE seed and convolution width (`true`, `1234`, `4`).
Small source/output receipts are in `config-receipts/`; the provenance file
hashes their original local paths. Both running nodes match the pinned FP8
loader and corrected weight checker sources.

The earlier native FP8 record verifies all 144 checkpoint files on both nodes,
plus exact equality of all checked common non-expert language tensors and FP8
PLE table payloads with NVFP4. No checkpoint download or conversion was needed.
The completed campaign restored all four original production binaries and the
resolved configuration, passed inference, collected the peer trace and exited 0.

## Excluded attempts

`attempt1.json`: the initial launch stopped at the idle guard without stopping
production or sending inference. The user subsequently authorized cluster tests.

`attempt2.json`: a diagnostic assertion incorrectly required FP32 checkpoint
scales. The actual scale tensors are BF16; the assertion stopped execution before
the first expert computation. No accuracy result is valid from that attempt.
Production restoration passed. The corrected checker requires the actual types
and exact promoted values. Its CPU preflight independently expands BF16 bits,
constructs full 128-by-128 scale blocks, shards and regroups them into 64-by-64
blocks. All 48 checks pass and corrupted scales are rejected on both ranks.
The successful retry requires that preflight and its checker hash.
