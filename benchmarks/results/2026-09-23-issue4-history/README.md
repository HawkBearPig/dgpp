# Issue #4: full-history replay and PLE EOS mismatch

The full 261,290-token forced-prefix capture completed on TP2 and preserved
exactly the clean run's prediction (`5`) and log probabilities. Production was
restored with its original binary hashes and resolved configuration, and an
independent HTTP smoke test passed. The capture source is retained in
`history.patch` and `issue4_history.hpp`.

The 80.85 GB raw capture set has since been relocated to node 13 at
`/home/stephen/dgpp-issue4-archive-20260923/history-captures`. All 1158 files
passed independent size/SHA256 verification before the local copy was removed.
The [relocation record](../2026-09-23-issue4-ple-quantization/capture-relocation.json)
and adjacent complete manifests retain its identity and location.

The independent PLE audit found a configuration defect: the trained model
uses EOS `248044`, but serving overwrites its model configuration with
`generation_config.json`'s stop list `[248046, 248044]`. Both PLE and session
initialization take the first ID. This changes 40 n-gram lookup IDs at prompt
positions 0, 1, 24 and 261112. The model EOS means `<|endoftext|>`; the generation
list's first entry means `<|im_end|>`. The model must not reset PLE at that chat
marker. The pinned independent vLLM PLE implementation also takes EOS from the
model config, not the generation stop list; source URLs/hashes are retained in
`raw/reference-source/sources.json`.

[The correction campaign](../2026-09-23-issue4-eos/README.md) tests causality.
This defect is established independently of whether correcting it resolves the
retrieval failure. [history-summary.json](history-summary.json) records the
full-history checks:

- 49,664 exact live invariants across 96 layer/rank combinations and 128 chunks:
  initial zero state, full GDN/PLE state handoffs and tails, logical K/V contents,
  old compressed-index entries, and chunk positions.
- All token IDs, positions and request IDs match the original diagnostic prefix.
  All final-layer residuals match the earlier operator capture bit for bit.
- 72 sampled GDN heads, one per recurrent layer/rank, independently replayed
  from zero with their own FP64 state for all 261,290 tokens. Maximum relative L2:
  convolution `2.417e-5`, recurrence output `1.496e-3`, checkpoint state `1.234e-3`.
  This samples arithmetic; it does not check every recurrent head's calculations.
- 24 complete compressed-index streams: maximum relative L2 `2.009e-5` against
  independent compression, norm and RoPE calculations.
- PLE convolution/residual calculations over all rows and channels on both ranks:
  maximum relative L2 `1.919e-7`. All runtime n-gram IDs and FP8 table gathers match
  the independently reconstructed *runtime* EOS policy bit for bit. That policy
  differs from the trained model as described above.
- No nonfinite values in the replays. No cache/state continuity defect detected.

`validate_history.py` first stopped on the model/runtime EOS mismatch. It now
reports that mismatch explicitly and separately checks all runtime lookups.
The numerical replay does not reseed recurrent state from captured chunk
boundaries. It does use captured operator inputs to isolate each calculation;
it is not a full independent model implementation.

The earlier residual-mixer replay follows.

`gr_replay.py` reconstructs the final-row attention mixer from checkpoint
BF16 weights, independently quantized to block FP8 with partial-block support,
and applies group normalization, down projection, activation, up projection
and gated mixing. Projections accumulate in FP64 with explicit BF16 staging.
The script does not invoke DGPP's mixer, norm, quantization or reference code.

It checks zero-based layer 0 against the actual token's checkpoint embedding,
and layers 2–47 against the preceding captured residual, on both ranks.
Layer 1 is excluded: the checkpoint's one-based PLE layer ID is 2, and PLE
changes the residual before this attention mixer. The script verifies that
the configuration's PLE layer set matches the checkpoint tensor names.

Executed results:

- 94 comparisons; no nonfinite values.
- Maximum relative L2: `0.0018211590` at layer 44.
- Maximum fraction exceeding two BF16 relative units with a `1e-7` floor:
  `0.012109375` at layer 10.
- 310 such scalar mismatches out of 240,640 compared scalars.
- These maxima are within the existing end-to-end mixer limits in
  `tests/cuda/qwen_gr_test.cu`: relative L2 `0.002`, mismatch fraction `0.02`.
  The reference here uses higher-precision dot products, so this is a
  numerical check, not a claim of identical scalar execution.

The first harness attempt incorrectly treated the checkpoint's PLE layer ID
as zero-based and compared across the intervening PLE update at layer 1.
That apparent large mismatch was a diagnostic-input error. The attempt is
retained under `raw/gr-invalid-ple-index`; only `raw/gr` is the corrected
result, with the omitted layer derived from and checked against the checkpoint.

This experiment still starts from captured production residuals. It does not
validate their earlier construction, GR injection, PLE, or a complete reference
model execution. By itself it does not select a fix. No production service or model source
was changed for this earlier CPU replay.

Replay from the repository root, choosing a new output directory:

```sh
OPENBLAS_NUM_THREADS=2 .venv/bin/python \
  benchmarks/results/2026-09-23-issue4-history/gr_replay.py \
  --captures benchmarks/results/2026-09-23-issue4-operators/raw/captures \
  --tokens benchmarks/results/2026-09-23-issue4-operators/raw/forced-prefix.ids.json \
  --checkpoint /home/stephen/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47 \
  --out /tmp/issue4-gr-replay-new
```

NumPy is required. The token count and published hash are checked before
replay. Raw results include checkpoint-tensor hashes; `provenance.json` binds
the script, corrected results, log and the prior capture manifest.
