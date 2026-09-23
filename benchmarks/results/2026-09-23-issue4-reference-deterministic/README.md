# Issue #4: deterministic reference QSA control

Executed twice on the original request. Both runs score 5/6 with the same wrong
value `val_5b9d2c3d7a95b986` and 190 completion tokens. Assistant text and usage
are identical. Both ranks each have 108120 equal captured fields and no differences.
Production restoration and peer capture collection both completed successfully.

This establishes repeatability in this two-request control, not universal
determinism or correctness. The two overlays remove the observed reference
variation but do not resolve the retrieval error.

The control Retains the non-fused expert finalization and its verified
cache from the previous control. Replaces QSA top-k with exact lexicographic
selection: score descending, lower index for equal scores, then ascending index
emission. Float32 scores are packed losslessly into int64 keys; no epsilon changes
scores. Padding follows live indices as required by the expansion kernel.

The smaller internal score batch (128 rows) bounds temporary key storage. The
original input, model, attention budget and prefill chunk size remain unchanged.
Host tests compare against independent NumPy lexsort at widths 512, 1024 and
65536, including ties, signed zero, infinities, empty and short visible prefixes.

Capture: full layer 0–3 boundaries in the first two chunks; last rows otherwise.
Read-only repeated original top-k calls save scores/selected sets at layer 3 in
those chunks, before using the deterministic result. Instrumentation can alter
scheduling, so this is a diagnostic reference control, not an uninstrumented run.

Related prior report: https://github.com/vllm-project/vllm/issues/54521#issuecomment-5480778284
DGPP already uses deterministic score/index selection. This does not itself
constitute a DGPP fix or prove the cause of issue #4.

Real repeated native top-k probes on rank 0: 8192 rows across the two requests,
4090 rows with changing output order, two row instances with a changing selected
set, zero invalid selections and zero selected-score errors against independent
NumPy ranking. Both set differences are the same prompt position, 3058: block
indices 214 and 498 tie exactly at score 4.2301926612854. This validates both
order and tie-selection variability on the actual fixture. The repeated score
inputs are bitwise identical. See `topk-rank0.json` and
`first-request-tie-events.json`.

Both-rank probe totals (`topk.json`): 16384 rows, 8180 with variable order,
two with variable sets, four instances of a tied threshold, zero invalid rows
and zero selected-score errors. All repeated score inputs match exactly.
`rank-comparison.json` records completed repeated-request comparisons.
