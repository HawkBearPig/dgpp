# Untraced deterministic W4A4 reference — running

Retry of the invalid explicit-backend comparison. This retains the original
`auto` backend selection and verifies both workers' runtime-selected cache
paths and cache hashes after initialization, before requesting inference.
The expected cache is the retained deterministic non-finalized W4A4 cache.

It removes tracing and read-only probes while preserving deterministic QSA
and non-fused MoE finalization. Two unchanged original requests are planned.
This separates the tracing difference from the successful untraced W4A16
reference. Production restoration runs in finally.

An initial startup stopped at a frozen-kernel harness API error before any TP2
reference world was started; logs and verified production restoration are
quarantined in `raw/invalid-unit-startup`. The corrected isolated comparison
completed, its container was removed, and reference TP2 initialization follows.
