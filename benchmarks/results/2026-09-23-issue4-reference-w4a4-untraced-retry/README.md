# Untraced deterministic W4A4 reference — complete

Retry of the invalid explicit-backend comparison. This retains the original
`auto` backend selection and verifies both workers' runtime-selected cache
paths and cache hashes after initialization, before requesting inference.
The expected cache is the retained deterministic non-finalized W4A4 cache.

It removes tracing and read-only probes while preserving deterministic QSA
and non-fused MoE finalization. Two unchanged original requests completed.
This separates the tracing difference from the successful untraced W4A16
reference. Production restoration completed with an inference smoke test and exact binary hashes on all four nodes.

An initial startup stopped at a frozen-kernel harness API error before any TP2
reference world was started; logs and verified production restoration are
quarantined in `raw/invalid-unit-startup`. The corrected isolated comparison
completed, its container was removed, and reference TP2 initialization follows.

A second startup was aborted by an overly strict log check before inference:
rank 1 reports the loaded cache through FlashInfer's loader, but does not emit
the rank-zero-only vLLM announcement. Both logs actually identify the expected
cache. That world and its verified restoration are retained under
`raw/aborted-cache-log-parser`. The corrected parser is checked against both
retained logs; the next attempt reuses the successful frozen-kernel receipt.

Both current workers loaded the expected retained cache and passed the exact
SHA256 check after initialization. `effective-cache.json` records this runtime
evidence. Both unchanged requests failed the same key, returning `val_5b9d2c3d7a95b986` (5/6, 190 completion tokens). Full assistant text and full usage match both traced deterministic reference requests. Cache hashes stayed unchanged after both requests. QSA, helper and PLE overlays match the successful Marlin world byte-for-byte. This removes tracing as an explanation for the observed W4A4/W4A16 accuracy difference; it does not identify the responsible arithmetic operation.
