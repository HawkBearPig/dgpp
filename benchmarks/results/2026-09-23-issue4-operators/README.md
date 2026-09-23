# Issue #4: independent attention and state investigation

The reporter's unpublished patch is **not a prerequisite** for investigating
the retrieval failure. It addresses a separately reported prefill consistency
problem and still predicts the wrong token. The existing evidence is enough
to reconstruct the diagnostic input and instrument current master ourselves.

This record continues the [original-request reproduction](../2026-09-23-issue4-exact/README.md).
The result remains unlocalized: the new boundary checks have not identified a
production defect or established that quantization/model behavior causes the
failure. No speculative production fix is proposed.

## Executed experiment

Base commit: `c6ca191863d620360f3b4370c0156f9e26a2631f`.
Checkpoint: `nvidia/Qwen3.8-Flash-Next-NVFP4`, snapshot
`fc694b54fb0174e0913e6adf86691ef85a4ead47`.

Using the original rendered prompt and saved assistant answer, we independently
reconstructed the 261,290-token prefix ending in the failed association's
`val_`. Its int64-LE token SHA256 exactly matches the reporter's published hash:
`fe207c412e6a70d70b00bbc3c5995488ed12ce964c4ac7eb90933ab61e82785d`.
`prepare_fixture.py` reproduces this reconstruction.

Two fresh TP2 worlds received this string through `/v1/completions`, each as its
first generation request. Both used native prefill budgets 0/0 and the original
checkpoint, BF16 KV and FP8 dense choices. MTP and prefix caching were disabled
for this single-token diagnostic. These controls differ from the separate
original free-generation experiment and are recorded in `forced.json`.

| Build | Executable SHA256 | Result |
| --- | --- | --- |
| Clean master | `b6afe5546afb687e4f9adea276625b2cbff2ac38ba6dc60eab55879a50ddba47` | `5`, logprob `-1.4839821` |
| Capture build | `0d5f7005eb7bff4aa2f9091fd4d05fba8b1fe49efb0ce41ea5b0ed94fce950ee` | Identical choices, logprobs and usage |

The expected next token is `a`. Both responses report 261,290 prompt tokens,
one completion token, zero cached tokens and a length finish, as expected for
`max_tokens: 1`. Top five: `5`, `7`, `3`, `9`, `0`. The winning margin over the
runner-up is approximately 0.47349 log-probability units.

Temporary instrumentation only copies tensors to host. It captured all 48
layers on both ranks for the final native chunk: positions 260,096–261,289,
1,194 rows. The diagnostic sources are retained as `diagnostic.patch` plus
`issue4_capture.hpp`, on branch `investigate/issue4-operator-capture` in
`/tmp/dgpp-issue4-20260923`. These sources hard-code the diagnostic position
and output path and are not intended for production integration.

## Independent numerical checks

`audit.cpp` uses scalar CPU arithmetic, mostly FP64, on production inputs.
It does not call DGPP kernels or its GDN/QSA reference implementations; it
shares only IEEE BF16 conversion helpers. It intentionally tests local
operator arithmetic, not whether the captured inputs are semantically right.

- **GDN, 72 layer/rank captures:** all 1,194 rows of convolution, recurrence
  and gated normalization, plus final convolution/recurrent state. Worst
  relative L2: convolution `4.40e-5`, recurrence `2.18e-4`, final recurrent
  state `1.75e-5`, gated normalization `3.92e-5`. Convolution-state placement
  is bitwise exact. No nonfinite values were found in the comparisons.
- **QSA, 24 layer/rank captures:** last-row query/index normalization and
  RoPE, the final chunk's 298 completed compressed pools, all 65,322 index
  score keys and all 2,050 selected token indices. Query/index normalization
  and RoPE are bitwise exact. Every score key matches an independent
  source-order FP32 calculation; selection also matches FP64 score ranking.
  Appended K/V cache tails are bitwise equal to their captured source arrays.
- **Attention arithmetic:** independent selected-K/V FP64 softmax/value
  accumulation has worst relative L2 `2.92e-4` against float kernel outputs.
  A separate BF16-unnormalized-exponential reference gives `4.14e-4` before
  gating and `1.65e-3` after gating. The latter stays inside the existing
  multi-split QSA tolerance (`4e-3` relative L2, eight BF16 relative units,
  2% RMS cancellation floor); no element exceeds the combined elementwise
  threshold. This is not bitwise equivalence to a different framework's
  softmax rounding order.
- **Cross-rank boundaries:** all 192 comparisons are bitwise equal: the
  attention inputs, folded attention outputs and final-row residual at all
  48 layers, plus QSA index query, cache, score and selection arrays at all
  12 sparse-attention layers.

`projections.py` separately reads BF16 checkpoint tensors and implements
block E4M3 quantization using NumPy arithmetic, independent of DGPP's loader
and FP8 decoder. It checks last-row GDN qkv/z/a/b, QSA query/index/value, and
all 48 output projections plus the TP2 fold, with FP64 dot-product accumulation
and BF16 staging. This covers 408 comparisons / 885,120 scalars. Worst relative
L2 is `4.7823e-4`; all GDN a/b projections are bitwise exact. Forty-two scalar
differences exceed a two-BF16-unit relative threshold with a `1e-7` floor;
all are below 2% of the reference tensor RMS. These measurements are retained,
not converted into a claim that every scalar agrees. No nonfinite values or
large projection error was observed. This does not cover the QSA key
projection directly, or the GR/PLE pathways.

`summary.json`, `projection-summary.json`, and the raw per-layer audit files
contain the complete measurements. `checkpoint-tensor-hashes.json` binds the
checkpoint tensors read by the projection replay.

## Retrieval observation and remaining gap

The correct ledger record is at tokens 258,280–258,314; the source of the
wrong value is at 251,069–251,097. Every QSA layer selects some tokens from
the correct record, from 3 to 27 tokens. Thus complete omission of that
record by every sparse selector at this row is not the explanation.
This says nothing about which record the model has understood or used.
`retrieval.json` records selected tokens and independent FP64 attention mass
per head for each target record.

The replay starts from **captured** inputs, old cache entries, and GDN state
at position 260,096. A defect accumulated earlier can pass every local check.
Likewise, model/checkpoint behavior and numerical differences can produce
an incorrect retrieval without a locally defective operator. The reporter's
vLLM W4A4 result is not a matched-W4A16 hidden-state oracle.

The next discriminating test is an independent reference of state/cache
construction from earlier in the prompt, preserving the same weights,
quantization and token IDs. Trace the first divergence instead of changing
kernel rounding to make this one answer pass. GR/PLE construction and earlier
projection rows remain uncovered by this campaign. A demonstrable mismatch
should become a focused regression before implementing a correction. Review
of the reporter's separate consistency patch can proceed independently.

## Replay and restoration

From the repository root, with retained captures present:

```sh
g++ -std=c++20 -O3 -ffp-contract=off -fopenmp -Isrc \
  benchmarks/results/2026-09-23-issue4-operators/audit.cpp \
  -o benchmarks/results/2026-09-23-issue4-operators/raw/audit
python3 benchmarks/results/2026-09-23-issue4-operators/audit_all.py
OPENBLAS_NUM_THREADS=1 python3 benchmarks/results/2026-09-23-issue4-operators/analyze.py
OPENBLAS_NUM_THREADS=2 python3 benchmarks/results/2026-09-23-issue4-operators/projections.py
```

The Python analysis requires NumPy; fixture reconstruction also requires
`tokenizers`. `projections.py` uses the recorded local snapshot path.
`raw/campaign.py`, command receipts, rank logs and resolved configurations
record the serial GPU campaign. The raw capture tree is about 5 GiB and is
ignored by Git. The diagnostic patch requires copying `issue4_capture.hpp`
into `src/models/qwen/` before building.

The original four-node GLM service was restored at 06:00 UTC. Direct readback
verified the original binary SHA256 on all four ranks, identical resolved
configuration, successful `OK` inference and idle status. See
`raw/restoration.json`. The diagnostic build was not left deployed, and the
user's original source checkout and executable were preserved.
