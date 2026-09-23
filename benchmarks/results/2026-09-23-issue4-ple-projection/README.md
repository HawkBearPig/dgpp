# PLE projection/gate and QSA key construction — completed; gate sensitivity localized

This audit closes two specific gaps in the earlier operator checks without
changing production inference arithmetic. CPU audits ran with production online;
a bounded GPU probe used the authorized stop/restore window. It does not establish
a retrieval fix. The original near-262K request remains unresolved.

The earlier full-history PLE audit independently validated lookup and convolution,
but accepted the already-computed gate and convolution-normalization inputs.
Here the PLE key/value projections, rank partials/fold, group normalizations and
gate are reconstructed independently from checkpoint weights and the captured
embedding/residual operands. The small replay covers 774 positions spanning all
128 native chunks, six target/source record ranges, EOS-sensitive positions and
the final query/forced-answer tail. The complete streaming replay covers all
261290 positions and checks 2675609600 scalars in each of three comparisons.
It finishes in 413.229 seconds with peak RSS 919928 KiB. All eight complete
source-file hashes match the verified archive; both ranks' replicated inputs
and outputs match exactly. The key/value projections and gate are now checked
through the entire captured prompt rather than only the final row.

The QSA audit covers all 1194 final-chunk rows at all 24 layer/rank pairs, from
checkpoint key projection through norm/RoPE to the cached keys. This key
projection was explicitly outside the earlier dense-projection audit's scope.
Its worst per-layer/rank relative L2 is 0.000241525 across 7335936 output
scalars. All cached tails equal their captured transform outputs exactly.
Numerical errors, including worst rows and relative outliers, are retained in
`qsa-keys.json`; this is not a claim of bitwise agreement with FP64 products.

## Evidence and limits

The CPU oracles do not call DGPP kernels or its host references. They implement
FP8 block conversion in NumPy, project using FP64 accumulation, and explicitly
round BF16 staging and TP2 partials/folds. The conversion preflight matches Torch
bitwise on 10005 BF16 values and 65536 block-FP8 values. It includes varying
scales, a zero block, subnormals and signed zero. Streaming error aggregation
matches whole-array statistics and detects two injected errors.

The source captures come from the original native NVFP4 forced-prefix diagnostic:
261290 tokens ending in the failed key's `val_`. That read-only capture preserved
the clean prediction and log probabilities. It predates the separate EOS fix;
the captured runtime EOS is 248046, while the trained EOS is 248044. This replay
intentionally starts from the actual saved embeddings and residuals, isolating
the downstream computations. It neither reintroduces nor claims to fix that
already-established configuration defect. The examined native operator sources
are unchanged between the capture base c6ca191 and current HEAD (`source-scope.json`).

The sampled replay's full checkpoint-to-gate relative L2 is 0.000575094;
checkpoint-to-convolution-normalization is 0.000247483, and isolated
convolution normalization is 0.0000116576. These describe numerical agreement
on captured inputs, not the correctness of the earlier computation that
created those inputs. No tolerance was adjusted to make this fixture pass.
No GPU accuracy run is replaced by these operator measurements.

## Reproduction

The source archive is on node 13 under
`/home/stephen/dgpp-issue4-archive-20260923/history-captures`; the full archive
passed independent size/SHA256 verification before relocation. Bounded exports
check source size and read stability and hash every exported payload. The full
stream also rehashes all eight source files as it reads them. Raw exported rows
and the reconstructed 131 MB weight bundle are retained locally, not committed.

1. Use `export_rows.py` with `selected-positions.json` and the retained full
   archive manifest to export the selected rows.
2. Run `check_oracle.py` with CPU NumPy/Torch.
3. Run `replay.py --rows raw/rows --checkpoint CHECKPOINT --output replay.json`.
4. Run `prepare_weights.py --checkpoint CHECKPOINT --output raw/weights.npz`.
5. On the archive host, run `stream_all.py --archive ARCHIVE --manifest MANIFEST
   --weights weights.npz --output full-replay.json --batch 512` with
   `OPENBLAS_NUM_THREADS=4`. NumPy runs in an isolated temporary package directory;
   no system Python packages or GPU configuration are changed.
6. Run `qsa_keys.py --captures OPERATOR_CAPTURE --checkpoint CHECKPOINT
   --output qsa-keys.json` against the retained 5 GiB operator capture.

The first full-stream attempt was deliberately stopped after 98816 rows because
mapped file pages grew process RSS to about 12.3 GiB. It is excluded from full
coverage. The retry replaces only the reader with fixed-size buffered reads;
its arithmetic is unchanged. `attempt1.json` retains the excluded receipt.

## Complete-stream result and localized outliers

Full-stream relative L2: checkpoint-to-gate **0.000256816**, checkpoint-to-conv
normalization **0.000218076**, isolated conv normalization **0.0000124758**.
No nonfinite values occur. Aggregate agreement hides local amplification:
position **136732** has gate-output relative L2 **0.0187652** (maximum absolute
error 0.000213623). The union of the 20 worst rows in the gate and normalization
chains contains 29 positions. All precede the final prompt chunk.

A CPU-only probe of the actual production FP8 encoder matches the independent
reconstruction bit-for-bit on **all 32768000 PLE projection weight values**.
FP32 projection accumulation and an explicit native-order FP32 reduction model
on CPU do not reproduce the outliers (`outliers.json`). No tolerance is relaxed.

## Frozen GPU attribution

`gpu_probe.cpp` preserves the native 2048-row shapes, strides, original row
indices and 64 MiB GEMM workspace, placing the 29 outlier operands at their
original positions and zero-padding other rows. It uses the actual DGPP GEMM,
norm and gate implementations from the validated diagnostic build. Both rank
projection partials are rounded/folded at the existing BF16 boundaries.

The native GPU replay **exactly reproduces all 296960 captured gate elements
and all 296960 captured convolution-normalization elements**. A second variant
uses FP32 GEMM output and rounds each rank partial back to BF16 at the original
boundary; it returns identical intermediate fields and outputs. It does not
keep the rank fold in FP32 and does not test FP64 accumulation.

Independent stage checks localize the large outliers upstream of the gate:

- The gate on actual GPU normalized key/query/value inputs is **bitwise exact**.
- Isolated key normalization differs in only 3 of 296960 values; applying its
  independent result to the actual query/value leaves gate outputs exact.
- The two key projection partials differ from the FP64 oracle in 190 and 164
  BF16 values; 118 values differ after the rank fold.
- At position 136732, the first gate is **0.03173828125** on GPU versus
  **0.0306396484375** from the oracle key projection, a seven-code BF16 shift.
  Replacing only the key chain in a fixed-input CPU intervention reproduces
  this change. Query normalization and value projection leave this row's
  gate scalars unchanged. All three other branch scalars agree.

This establishes numerical amplification of key-projection differences on
actual operands. It does **not** yet establish a defective GEMM, a violation
of its numerical contract, or causality for the incorrect retrieved value.
The next discriminating experiment is a validated higher-accuracy **PLE key
projection only**, followed by the unchanged original request on real TP2.
A production change is warranted only if that experiment establishes a repair.

The bounded GPU campaign stopped production at 18:19:49 UTC; both probes ran
successfully in about two seconds total, and production restoration completed
with all four original executable hashes, exact configuration, `OK` inference
and idle/nonfailed service state. `gpu-restoration.json` records the checks.
No retrieval fix or model-serving diagnostic was left deployed.
